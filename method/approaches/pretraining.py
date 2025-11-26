import torch
from functools import partial

from torch.optim.adamw import AdamW

from method import utils
from method.engine import train_and_eval


def train(model: torch.nn.Module, datasets, run, args):
    assert args.world_size == 1, "Pretraining is not compatible with distributed training"
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Not training {name}")

    train_dataset = (
        datasets.get_train_data(subsample=args.rob_approach == "subsample")
        if args.method == "pretrain"
        else datasets.get_unlearning_data(train=True, subsample=args.rob_approach == "subsample")[
            "retain"
        ]
    )
    val_dataset = datasets.get_val_data()

    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )

    # if dro is activated, compute dro weights for the weighted sampler
    sampler_weights = None
    num_groups = 0
    num_sensitive_attr = 0
    if args.rob_approach in ("dro", "target_reweight"):
        if args.rob_approach == "target_reweight":
            from method.approaches.target_reweight import compute_weights
        else:
            from method.approaches.dro import compute_weights

        sampler_weights, group_counts, num_sensitive_attr, num_groups = compute_weights(
            datasets, split="train" if args.method == "pretrain" else "retain"
        )

    # get samplers
    train_sampler = utils.get_sampler(train_dataset, shuffle=True, weights=sampler_weights)
    val_sampler = utils.get_sampler(val_dataset, shuffle=False)

    train_loader = generic_loader(train_dataset, sampler=train_sampler)
    val_loader = generic_loader(val_dataset, sampler=val_sampler)

    criterion = utils.get_criterion(
        args.criterion,
        dro=args.rob_approach,
        dro_lr=args.dro_lr,
        num_groups=num_groups,
        num_sensitive_attr=num_sensitive_attr,
    ).to(args.device)

    if args.model != "bert":
        optimizer = utils.get_optimizer(model, args)
        scheduler = utils.get_scheduler(optimizer, args)
        warmup_scheduler = utils.get_warmup_scheduler(optimizer, len(train_loader), args)

    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_params,
            lr=args.lr,
        )
        scheduler = utils.get_scheduler(optimizer, args)
        warmup_scheduler = utils.get_warmup_scheduler(optimizer, len(train_loader), args)

    utils.print_info(args, model, train_loader)

    best_model, last_model = train_and_eval(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        clip_grad=args.clip_grad,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        debug=args.debug,
        run=run,
        task=args.task,
        evaluate_every=args.evaluate_every,
        args=args,
    )

    return best_model


def pretrain(model, datasets, run, args):
    return train(model, datasets, run, args)


def retrain(model, datasets, run, args):
    return train(model, datasets, run, args)
