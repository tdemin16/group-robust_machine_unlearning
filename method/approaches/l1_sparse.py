import copy
import json
import os
import time
import torch

from method import utils
from method.metrics import evaluate_after_unlearning, compute_equalized_odds


def compute_alpha(curr_epoch, epochs, alpha):
    return alpha * (1 - curr_epoch / epochs)


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def fine_tune(model, datasets, run, args):
    assert args.world_size == 1, "L1-FT-GA are not compatible with distributed training"
    assert (
        args.task == "classification"
    ), "L1-FT-GA are not compatible with multilabel classification"

    if args.rob_approach != "none" and args.method == "ga":
        args.rob_approach = "none"
        print(
            f"{utils.bcolors.WARNING}[WARNING]{utils.bcolors.ENDC} rob_approach flag is not set to none but GA does not compute gradient descent on retain data. Flag will be ignored..."
        )

    unlearning_datasets = datasets.get_unlearning_data(
        train=args.use_train_aug, subsample=args.rob_approach == "subsample"
    )
    if args.method == "ga":
        dataset = unlearning_datasets["forget"]
    else:
        dataset = unlearning_datasets["retain"]

    # if dro is activated, compute dro weights for the weighted sampler
    dro_weights = None
    num_groups = 0
    num_sensitive_attr = 0
    if args.rob_approach in ("dro", "target_reweight"):
        if args.rob_approach == "target_reweight":
            # from method.approaches.target_dro import compute_weights
            #! IMPLEMENT
            pass
        else:
            from method.approaches.dro import compute_weights

        dro_weights, group_counts, num_sensitive_attr, num_groups = compute_weights(datasets, split="retain")

    sampler = utils.get_sampler(dataset, shuffle=True, weights=dro_weights)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    utils.print_info(args, model, loader)

    unlearned_model = copy.deepcopy(model)
    unlearned_model.to(args.device)

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
        alpha = compute_alpha(epoch, args.epochs, args.l1_penalty)
        for image, target, sensitive in loader:
            with torch.autocast(device_type="cuda", dtype=args.dtype):
                image = image.to(device=args.device, dtype=args.dtype)
                target = target.to(args.device)
                sensitive = sensitive.to(args.device)

                output = unlearned_model(image)
                if "dro" not in args.rob_approach:
                    loss = criterion(output, target)
                else:
                    loss = criterion(output, target, sensitive)

                if args.method == "ga":
                    loss *= -1
                elif args.method == "l1_sparse":
                    loss += alpha * l1_regularization(unlearned_model)

            loss.backward()
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


def ft(model, datasets, run, args):
    return fine_tune(model, datasets, run, args)


def ga(model, datasets, run, args):
    return fine_tune(model, datasets, run, args)


def l1_sparse(model, datasets, run, args):
    return fine_tune(model, datasets, run, args)
