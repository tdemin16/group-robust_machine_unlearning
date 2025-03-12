import copy
import json
import math
import os
import time
import torch
from functools import partial
from torch.nn import functional as F
from tqdm import tqdm

from method import utils
from method.approaches.target_reweight import compute_weights
from method.metrics import evaluate_after_unlearning, compute_equalized_odds


class MINE(torch.nn.Module):
    def __init__(self, z_size: int, c_size: int, hidden_size: int = 512):
        super().__init__()
        self.c_size = c_size
        input_size = z_size + c_size
        self.neural_estimator = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, z, c):
        c = F.one_hot(c, num_classes=self.c_size)
        z_c = torch.cat((z, c), dim=-1)
        return self.neural_estimator(z_c)


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / (running_mean + 1e-18) / input.shape[0]
        return grad, None


class MutualInformation(torch.nn.Module):
    def __init__(self, mine: MINE, alpha=0.01):
        super().__init__()
        self.mine = mine
        self.alpha = alpha
        self.running_mean = None

    def forward(self, z, c, c_tilde):
        joint = self.mine(z, c).mean()
        x = self.mine(z, c_tilde)

        t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.size(0))).detach()

        if self.running_mean is None:
            self.running_mean = t_exp
        else:
            self.running_mean = self.alpha * t_exp + (1 - self.alpha) * self.running_mean.item()
        t_log = EMALoss.apply(x, self.running_mean)

        mi = joint - t_log
        return mi

    def train(self, mode: bool = True):
        super().train(mode)
        self.mine.train(mode)

    def eval(self):
        super().eval()
        self.mine.eval()


def tune_mine(model, mi, train_loader, num_sensitive_attr, optim, iterations, args):
    # register forward hook to store features
    def hook_fn(module, input, output):
        z.append(input[0])

    hook_handle_unlearned = model.fc.register_forward_hook(hook_fn)

    # get mi and optimizer for original and unlearning model
    mi_unlearning = mi["unlearning"]
    mi_original = mi["original"]
    optim_unlearning = optim["unlearning"]
    optim_original = optim["original"]

    model.train()
    mi_unlearning.train()
    mi_original.train()

    # should be useless
    optim_unlearning.zero_grad()
    optim_original.zero_grad()

    i = -1
    bar = tqdm(range(iterations), leave=False, dynamic_ncols=True, desc="Tuning MINE")
    for _ in range(iterations // len(train_loader) + 1):
        for image, target, sensitive in train_loader:
            z = []
            i += 1
            with torch.autocast(device_type="cuda", dtype=args.dtype):
                image = image.to(device=args.device, dtype=args.dtype)
                target = target.to(args.device)
                sensitive = sensitive.to(args.device)

                # random sensitive and target attributes (for marginalization)
                target_tilde = torch.randint_like(target, args.num_classes)
                sensitive_tilde = torch.randint_like(sensitive, num_sensitive_attr)

                # crompute group and random group (for marginalization)
                group = sensitive + target * num_sensitive_attr
                group_tilde = sensitive_tilde + target_tilde * num_sensitive_attr

                with torch.no_grad():
                    _ = model(image)

            # maximize mutual information
            loss = -mi_unlearning(z[0].float(), group, group_tilde)
            loss += -mi_original(z[0].float(), group, group_tilde)

            if torch.isnan(loss):
                print(f"Loss is {loss.item()}.")
                exit()

            loss.backward()
            optim_unlearning.step()
            optim_original.step()
            optim_unlearning.zero_grad()
            optim_original.zero_grad()

            bar.update(1)
            if i == iterations or args.debug:
                break

        if i == iterations or args.debug:
            break

    hook_handle_unlearned.remove()


def grad_norm(module):
    params = [p for p in module.parameters() if p.grad is not None]
    norm = sum([p.grad.data.norm(2).item() ** 2 for p in params]) ** 0.5
    return norm


def gradient_clipping(model, mine):
    model_norm = grad_norm(model)
    mine_norm = grad_norm(mine)

    min_norm = min(model_norm, mine_norm)
    params = [p for p in mine.parameters() if p.grad is not None]

    for p in params:
        p.grad.data *= min_norm / (mine_norm + 1e-18)


def print_group_dist(dataset, num_sensitive_attr):
    counts = {}
    for i in range(len(dataset)):
        _, target, sensitive = dataset[i]
        group = sensitive + target * num_sensitive_attr
        if str(group) not in counts:
            counts[str(group)] = 0
        
        counts[str(group)] += 1

    print(counts)
    exit()


def miu(model, datasets, run, args):
    assert args.world_size == 1
    assert args.task == "classification"
    assert args.rob_approach == "none"

    # create forget and retain dataloaders
    train_data = datasets.get_train_data()
    unlearning_datasets = datasets.get_unlearning_data(train=args.use_train_aug)
    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    forget_loader = generic_loader(unlearning_datasets["forget"], shuffle=True)
    train_loader = generic_loader(train_data, shuffle=True)

    # get sampler weights, number of sensitive attr to compute group, and number of groups
    balanced_weights, group_counts, num_sensitive_attr, num_groups = compute_weights(
        datasets, split="retain"
    )

    # print_group_dist(unlearning_datasets["forget"], num_sensitive_attr)

    # reweight retain sampler if in control bias mode
    sampler_weights = None
    if args.control_bias and not args.ablate_rw:
        sampler_weights = balanced_weights
        print("Using reweighting...")
    retain_loader = generic_loader(
        unlearning_datasets["retain"],
        sampler=utils.get_sampler(
            unlearning_datasets["retain"], shuffle=True, weights=sampler_weights
        ),
    )

    # make a copy of the model to unlearn
    unlearned_model = copy.deepcopy(model)
    unlearned_model.to(args.device)

    # unlearned model scheduler and optimizer
    model_optimizer = utils.get_optimizer(unlearned_model, args)
    model_scheduler = utils.get_scheduler(model_optimizer, args)

    # mine features-group to compute unlearning and debiasing
    mine_unlearning = MINE(z_size=512, c_size=args.num_classes * num_sensitive_attr)
    mi_unlearning = MutualInformation(mine_unlearning)
    mi_unlearning.to(args.device)
    mine_unlearning_optimizer = torch.optim.SGD(
        mine_unlearning.parameters(), lr=args.mine_lr, momentum=0.9
    )

    # mine features-attributes of original model
    mine_original = MINE(z_size=512, c_size=args.num_classes * num_sensitive_attr)
    mi_original = MutualInformation(mine_original)
    mi_original.to(args.device)
    mine_original_optimizer = torch.optim.SGD(
        mine_original.parameters(), lr=args.mine_lr, momentum=0.9
    )

    # merge objects for easier handling
    mi = {"unlearning": mi_unlearning, "original": mi_original}
    mine_optimizers = {"unlearning": mine_unlearning_optimizer, "original": mine_original_optimizer}

    utils.print_info(args, model, train_loader)

    acc_curves = {"UA": [], "TA": [], "GA": []}
    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        # start time count
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        unlearned_model.train()

        # register forward hook to store features
        z = []

        def hook_fn(module, input, output):
            z.append(input[0])

        hook_handle_unlearned = unlearned_model.fc.register_forward_hook(hook_fn)

        # add hook to original model if control bias
        if args.control_bias:
            hook_handle_model = model.fc.register_forward_hook(hook_fn)

        len_dataloaders = len(retain_loader)
        if epoch < 5:
            len_dataloaders += len(forget_loader)

        bar = tqdm(range(len_dataloaders), leave=False, dynamic_ncols=True)

        # Tune original and unlearning MINE for 100 iterations (batches)
        tune_mine(
            unlearned_model,
            mi,
            train_loader,
            num_sensitive_attr,
            mine_optimizers,
            100 if epoch == 0 else 10,
            args,
        )

        # unlearn for the first 5 epochs
        if epoch < args.forgetting_epochs and not args.ablate_unlearn:
            for image, target, sensitive in forget_loader:
                with torch.autocast(device_type="cuda", dtype=args.dtype):
                    image = image.to(device=args.device, dtype=args.dtype)
                    target = target.to(args.device)
                    sensitive = sensitive.to(args.device)

                    # compute group and random group for marginalization
                    group = sensitive + target * num_sensitive_attr
                    group_tilde = torch.randint_like(group, num_groups)

                    z = []
                    _ = unlearned_model(image)

                # minimize mutual information
                loss = mi_unlearning(z[0].float(), group, group_tilde)

                if torch.isnan(loss):
                    print(f"Loss is {loss.item()}.")
                    exit()

                loss.backward()
                gradient_clipping(model, mine_unlearning)
                model_optimizer.step()
                model_optimizer.zero_grad()

                bar.set_description("Forget")
                bar.update(1)

                if args.debug:
                    break

        for image, target, sensitive in retain_loader:
            with torch.autocast(device_type="cuda", dtype=args.dtype):
                image = image.to(device=args.device, dtype=args.dtype)
                target = target.to(args.device)
                sensitive = sensitive.to(args.device)

                # compute group and random group for marginalization
                group = sensitive + target * num_sensitive_attr
                group_tilde = torch.randint_like(group, num_groups)

                z = []
                output = unlearned_model(image)

            loss = 0
            if not args.ablate_retain:
                loss = F.cross_entropy(output, target).float()

            # mutual information alignment
            if args.control_bias and not args.ablate_alignment:
                feats = z[0].float()
                z = []
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=args.dtype):
                    _ = model(image)

                # original model features
                original_feats = z[0].float()

                # mutual information discrepancy
                loss += args.lambda_mi * F.mse_loss(
                    mi_unlearning(feats, group, group_tilde),
                    mi_original(original_feats, group, group_tilde).detach(),
                )

            if torch.isnan(loss):
                print(f"Loss is {loss.item()}.")
                exit()

            loss.backward()
            if args.control_bias and not args.ablate_alignment:
                gradient_clipping(model, mine_unlearning)
            model_optimizer.step()
            model_optimizer.zero_grad()

            bar.set_description("Retain")
            bar.update(1)

            if args.debug:
                break

        hook_handle_unlearned.remove()
        if args.control_bias:
            hook_handle_model.remove()

        bar.close()

        end.record()
        torch.cuda.synchronize()

        if epoch % args.evaluate_every == 0:
            (
                retain_loss,
                retain_acc,
                retain_losses,
                forget_loss,
                forget_acc,
                forget_losses,
                val_loss,
                val_acc,
                val_losses,
                test_loss,
                test_acc,
                test_losses,
            ) = evaluate_after_unlearning(
                unlearned_model, datasets, utils.get_criterion(args.criterion), args=args
            )
            print(
                f"| Epoch: {str(epoch+1).zfill(num_digits)}/{args.epochs} | LR: {model_optimizer.param_groups[0]['lr']:.4f} | Retain Loss: {retain_loss:.4f} | Retain Acc: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget Acc: {forget_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} | Time: {start.elapsed_time(end) / 1000:.1f} |"
            )
            if args.store_curves:
                eo, ga = compute_equalized_odds(unlearned_model, datasets, args)
                acc_curves["UA"].append(forget_acc)
                acc_curves["TA"].append(test_acc)
                acc_curves["GA"].append(ga)

        model_scheduler.step()

        if args.debug:
            break
    
    if args.store_curves:
        curves_dir = os.path.join(args.curves_store_dir, args.dataset, args.model)
        if not os.path.exists(curves_dir):
            try:
                os.makedirs(curves_dir)
            except OSError:
                time.sleep(5)
        curves_name = f"{args.method}_{args.seed}"
        curves_path = os.path.join(curves_dir, f"{curves_name}.json")
        print(f"Storing curves in {curves_path}")

        with open(curves_path, "w") as fp:
            json.dump(acc_curves, fp)


    return unlearned_model