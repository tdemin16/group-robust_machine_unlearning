import copy
import json
import os
import time
import torch
from functools import partial
from tqdm import tqdm

from method import utils
from method.metrics import evaluate_after_unlearning, compute_equalized_odds


def kl_div(prediction, target, temperature):
    prediction = torch.log_softmax(prediction / temperature, dim=1)
    target = torch.softmax(target / temperature, dim=1)

    return (
        torch.nn.functional.kl_div(prediction, target, reduction="sum")
        * (temperature**2)
        / prediction.size(0)
    )


def scrub(model, datasets, run, args):
    assert args.world_size == 1, "SCRUB is not compatible with distributed training"

    train_data = datasets.get_train_data()
    unlearning_datasets = datasets.get_unlearning_data(
        train=args.use_train_aug, subsample=args.rob_approach == "subsample"
    )
    retain_dataset = unlearning_datasets["retain"]
    forget_dataset = unlearning_datasets["forget"]

    generic_loader = partial(
        torch.utils.data.DataLoader,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )

    # if rob_approach is set, compute weights for the weighted sampler
    sampler_weights = None
    num_groups = 0
    num_sensitive_attr = 0
    if args.rob_approach in ("dro", "target_reweight"):
        if args.rob_approach == "target_reweight":
            from method.approaches.target_reweight import compute_weights
        else:
            from method.approaches.dro import compute_weights

        sampler_weights, group_counts, num_sensitive_attr, num_groups = compute_weights(
            datasets, split="retain"
        )

    train_sampler = utils.get_sampler(train_data, shuffle=False, weights=None)
    retain_sampler = utils.get_sampler(retain_dataset, shuffle=True, weights=sampler_weights)
    forget_sampler = utils.get_sampler(forget_dataset, shuffle=True, weights=None)

    train_loader = generic_loader(
        train_data, batch_size=args.batch_size_retain, sampler=train_sampler
    )
    retain_loader = generic_loader(
        retain_dataset, batch_size=args.batch_size_retain, sampler=retain_sampler
    )
    forget_loader = generic_loader(
        forget_dataset, batch_size=args.batch_size, sampler=forget_sampler
    )

    utils.print_info(args, model, train_loader)

    criterion = utils.get_criterion(
        args.criterion,
        dro=args.rob_approach,
        dro_lr=args.dro_lr,
        num_groups=num_groups,
        num_sensitive_attr=num_sensitive_attr,
    ).to(args.device)

    unlearned_model = copy.deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)

    optimizer = utils.get_optimizer(unlearned_model, args)
    scheduler = utils.get_scheduler(optimizer, args)

    unlearned_model.train()
    optimizer.zero_grad()

    acc_curves = {"UA": [], "TA": [], "GA": []}
    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        unlearned_model.train()
        model.eval()

        # ? Forget step
        if epoch < args.forgetting_epochs:
            for input, target, sensitive in tqdm(
                forget_loader, desc="Forget Step", leave=False, dynamic_ncols=True
            ):
                if args.model != "bert":
                    with torch.autocast(device_type="cuda", dtype=args.dtype):
                        input = input.to(device=args.device, dtype=args.dtype)
                        target = target.to(args.device)
                        sensitive = sensitive.to(args.device)
                        output = unlearned_model(input)
                        with torch.no_grad():
                            original_output = model(input)

                else:
                    input = input.to(device=args.device)
                    target = target.to(args.device)
                    sensitive = sensitive.to(args.device)
                    output = unlearned_model(
                        input_ids=input[:, :, 0],
                        attention_mask=input[:, :, 1],
                        token_type_ids=input[:, :, 2],
                        labels=target,
                        # return logits
                    )[1]
                    with torch.no_grad():
                        original_output = model(
                            input_ids=input[:, :, 0],
                            attention_mask=input[:, :, 1],
                            token_type_ids=input[:, :, 2],
                            labels=target,
                            # return logits
                        )[1]

                loss = -kl_div(output, original_output, args.temperature_scrub)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if args.debug:
                    break

        # ? Retain step
        for input, target, sensitive in tqdm(
            retain_loader, desc="Retain Step", leave=False, dynamic_ncols=True
        ):
            if args.model != "bert":
                with torch.autocast(device_type="cuda", dtype=args.dtype):
                    input = input.to(device=args.device, dtype=args.dtype)
                    target = target.to(args.device)
                    sensitive = sensitive.to(args.device)
                    output = unlearned_model(input)
                    with torch.no_grad():
                        original_output = model(input)
            
            else:
                input = input.to(device=args.device)
                target = target.to(args.device)
                sensitive = sensitive.to(args.device)
                output = unlearned_model(
                    input_ids=input[:, :, 0],
                    attention_mask=input[:, :, 1],
                    token_type_ids=input[:, :, 2],
                    labels=target,
                    # return logits
                )[1]
                with torch.no_grad():
                    original_output = model(
                        input_ids=input[:, :, 0],
                        attention_mask=input[:, :, 1],
                        token_type_ids=input[:, :, 2],
                        labels=target,
                        # return logits
                    )[1]

            loss = args.alpha_scrub * kl_div(output, original_output, args.temperature_scrub)
            if "dro" not in args.rob_approach:
                loss += args.gamma_scrub * criterion(output, target)
            else:
                loss += args.gamma_scrub * criterion(output, target, sensitive)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.debug:
                break

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
            ) = evaluate_after_unlearning(unlearned_model, datasets, criterion, args=args)
            print(
                f"| Epoch: {str(epoch+1).zfill(num_digits)}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.4f} | Retain Loss: {retain_loss:.4f} | Retain Acc: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget Acc: {forget_acc:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f} | Time: {start.elapsed_time(end) / 1000:.1f} |"
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
