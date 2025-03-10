import copy
import json
import os
import random
import time
import torch
from functools import partial
from tqdm import tqdm

from method import utils
from method.metrics import evaluate_after_unlearning, compute_equalized_odds


def compute_mask(model, forget_loader, criterion, args):
    gradients = {}
    model.eval()
    model.zero_grad()

    for name, param in model.named_parameters():
        gradients[name] = 0

    print("Computing Gradients...")
    for batch in forget_loader:
        image = batch[-3]
        target = batch[-2]

        image = image.to(args.device)
        target = target.to(args.device)

        output = model(image)
        loss = -criterion(output, target)

        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

        model.zero_grad()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    mask = {}
    all_elements = -torch.cat([w.flatten() for w in gradients.values()])

    # calculate number of elements to keep
    threshold_index = int(len(all_elements) * args.saliency_threshold)

    # calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    print("Computing Saliency Mask...")
    start_index = 0
    for key, w in gradients.items():
        num_elements = w.numel()
        weight_ranks = ranks[start_index : start_index + num_elements]

        # set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(weight_ranks)
        threshold_tensor[weight_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(w.shape)
        mask[key] = threshold_tensor
        start_index += num_elements

    return mask


def random_labeling_big(model, datasets, use_mask, run, args):
    assert args.world_size == 1, "SalUn is not compatible with distributed training"
    assert args.task == "classification", "SalUn is not compatible with multilabel classification"

    train_data = datasets.get_train_data(
        args.use_train_aug, subsample=args.rob_approach == "subsample"
    )
    forget_dataset = datasets.get_unlearning_data(train=args.use_train_aug)["forget"]
    forget_indices = set(forget_dataset.indices)
    forget_targets = {i: random.randint(0, args.num_classes - 1) for i in forget_dataset.indices}

    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    # if dro is activated, compute dro weights for the weighted sampler
    dro_weights = None
    num_groups = 0
    sensitive_attr = []
    if args.rob_approach == "dro":
        from method.approaches import compute_weights  # avoids circular imports

        target_attr, sensitive_attr, num_groups = datasets.get_attrs(split="train")
        dro_weights = compute_weights(target_attr, sensitive_attr)

    train_sampler = utils.get_sampler(train_data, shuffle=True, weights=dro_weights)
    forget_sampler = utils.get_sampler(forget_dataset, shuffle=True, weights=None)

    train_loader = generic_loader(train_data, sampler=train_sampler)
    forget_loader = generic_loader(forget_dataset, sampler=forget_sampler)

    utils.print_info(args, model, train_loader)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    if use_mask:
        mask = compute_mask(model, forget_loader, criterion, args)

    unlearned_model = copy.deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)

    criterion = utils.get_criterion(
        args.criterion,
        dro=args.rob_approach == "dro",
        dro_lr=args.dro_lr,
        num_groups=num_groups,
        num_sensitive_attr=len(set(sensitive_attr)),
    ).to(args.device)

    optimizer = utils.get_optimizer(unlearned_model, args)
    scheduler = utils.get_scheduler(optimizer, args)

    acc_curves = {"UA": [], "TA": [], "GA": []}
    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        unlearned_model.train()

        # enable returning indices of training samples
        train_loader.dataset.dataset.return_indexes = True

        for indices, image, target, sensitive in tqdm(
            train_loader, leave=False, dynamic_ncols=True
        ):
            with torch.autocast(device_type="cuda", dtype=args.dtype):
                image = image.to(device=args.device, dtype=args.dtype)
                target = target.to(args.device)
                sensitive = sensitive.to(args.device)

                # use random labeling for forget data
                forget_samples = [
                    i for i, idx in enumerate(indices) if idx.item() in forget_indices
                ]
                if len(forget_samples) > 0:
                    forget_samples = torch.tensor(forget_samples)
                    random_targets = torch.tensor(
                        [
                            forget_targets[idx.item()]
                            for idx in indices
                            if idx.item() in forget_indices
                        ],
                        device=args.device,
                    )
                    target[forget_samples] = random_targets

                output = unlearned_model(image)
                if args.rob_approach != "dro":
                    loss = criterion(output, target)
                else:
                    loss = criterion(output, target, sensitive)

            loss.backward()

            if use_mask:
                for name, param in unlearned_model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            if args.debug:
                break

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
            ) = evaluate_after_unlearning(unlearned_model, datasets, criterion, args=args)
            print(
                f"| Epoch: {str(epoch+1).zfill(num_digits)}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.4f} | Retain Loss: {retain_loss:.4f} | Retain Acc: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget Acc: {forget_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} |"
            )
            if args.store_curves:
                eo, ga = compute_equalized_odds(unlearned_model, datasets, args)
                acc_curves["UA"].append(forget_acc)
                acc_curves["TA"].append(test_acc)
                acc_curves["GA"].append(ga)

        scheduler.step()

        if args.debug:
            break
    
    if args.store_curves:
        unlearning_str = args.unlearning_type
        if args.unlearning_type == "bias" and args.multi_group == 0:
            unlearning_str += f"_{args.split_number}"
        elif args.unlearning_type == "bias" and args.multi_group != 0:
            unlearning_str += f"_group{args.multi_group}"
        curves_dir = os.path.join(args.curves_store_dir, args.dataset, args.model, unlearning_str)
        if not os.path.exists(curves_dir):
            try:
                os.makedirs(curves_dir)
            except OSError:
                time.sleep(5)
        curves_name = f"{args.method}"
        if args.rob_approach != "none":
            curves_name = args.rob_approach + "_" + curves_name
        if args.method == "miu" and args.ablate_rw:
            curves_name = "rw_" + curves_name
        if args.method == "miu" and args.ablate_alignment:
            curves_name = "align_" + curves_name
        if args.method == "miu" and args.ablate_unlearn:
            curves_name = "ul_" + curves_name
        if (
            args.method != "pretrain"
            and args.method != "retrain"
            and args.pretrain_name != "pretrain"
        ):
            curves_name += f"_{args.pretrain_name}"
        curves_name += f"_{args.unlearning_ratio}"
        curves_path = os.path.join(curves_dir, f"{curves_name}.json")
        print(f"Storing curves in {curves_path}")

        with open(curves_path, "w") as fp:
            json.dump(acc_curves, fp)

    return unlearned_model


def random_labeling_small(model, datasets, use_mask, run, args):
    assert args.world_size == 1, "SalUn is not compatible with distributed training"
    assert args.task == "classification", "SalUn is not compatible with multilabel classification"

    train_dataset = datasets.get_train_data()
    unlearning_datasets = datasets.get_unlearning_data(
        train=args.use_train_aug, subsample=args.rob_approach == "subsample"
    )
    retain_dataset = unlearning_datasets["retain"]
    forget_dataset = unlearning_datasets["forget"]

    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    # if dro is activated, compute dro weights for the weighted sampler
    dro_weights = None
    num_groups = 0
    num_sensitive_attr = 0
    if args.rob_approach in ("dro", "target_reweight"):
        if args.rob_approach == "target_reweight":
            # from method.approaches.target_dro import compute_weights
            #! implement
            pass
        else:
            from method.approaches.dro import compute_weights

        dro_weights, group_counts, num_sensitive_attr, num_groups = compute_weights(
            datasets, split="retain"
        )

    train_sampler = utils.get_sampler(train_dataset, shuffle=False, weights=None)
    retain_sampler = utils.get_sampler(retain_dataset, shuffle=True, weights=dro_weights)
    forget_sampler = utils.get_sampler(forget_dataset, shuffle=True, weights=None)

    train_loader = generic_loader(train_dataset, sampler=train_sampler)
    retain_loader = generic_loader(retain_dataset, sampler=retain_sampler)
    forget_loader = generic_loader(forget_dataset, sampler=forget_sampler)

    utils.print_info(args, model, train_loader)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    if use_mask:
        mask = compute_mask(model, forget_loader, criterion, args)

    unlearned_model = copy.deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)

    criterion = utils.get_criterion(
        args.criterion,
        dro=args.rob_approach,
        dro_lr=args.dro_lr,
        num_groups=num_groups,
        num_sensitive_attr=num_sensitive_attr,
    ).to(args.device)

    optimizer = utils.get_optimizer(unlearned_model, args)
    scheduler = utils.get_scheduler(optimizer, args)

    acc_curves = {"UA": [], "TA": [], "GA": []}
    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        unlearned_model.train()
        for desc, loader in zip(["Forget", "Retain"], [forget_loader, retain_loader]):
            for image, target, sensitive in tqdm(
                loader, desc=f"{desc} Step", leave=False, dynamic_ncols=True
            ):
                with torch.autocast(device_type="cuda", dtype=args.dtype):
                    image = image.to(device=args.device, dtype=args.dtype)
                    if desc == "Forget":
                        target = torch.randint(0, args.num_classes, target.size())
                    target = target.to(args.device)
                    sensitive = sensitive.to(args.device)

                    output = unlearned_model(image)
                    if "dro" not in args.rob_approach:
                        loss = criterion(output, target)
                    else:
                        loss = criterion(output, target, sensitive)

                loss.backward()

                if use_mask:
                    for name, param in unlearned_model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                optimizer.step()
                optimizer.zero_grad()

                if args.debug:
                    break

        if args.debug:
            break

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
            ) = evaluate_after_unlearning(unlearned_model, datasets, criterion, args=args)
            print(
                f"| Epoch: {str(epoch+1).zfill(num_digits)}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.4f} | Retain Loss: {retain_loss:.4f} | Retain Acc: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget Acc: {forget_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} |"
            )
            if args.store_curves:
                eo, ga = compute_equalized_odds(unlearned_model, datasets, args)
                acc_curves["UA"].append(forget_acc)
                acc_curves["TA"].append(test_acc)
                acc_curves["GA"].append(ga)

        scheduler.step()

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


def rl(model, datasets, run, args):
    random_labeling = random_labeling_small if args.dataset != "cifar100" else random_labeling_big
    return random_labeling(model, datasets, use_mask=False, run=run, args=args)


def salun(model, datasets, run, args):
    random_labeling = random_labeling_small if args.dataset != "cifar100" else random_labeling_big
    return random_labeling(model, datasets, use_mask=True, run=run, args=args)
