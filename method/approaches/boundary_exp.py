import copy
import torch
from tqdm import tqdm

from method import utils
from method.metrics import evaluate_after_unlearning


def expand_model(model):
    last_fc_name = None
    last_fc_layer = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_fc_name = name
            last_fc_layer = module

    print(f"{utils.bcolors.OKBLUE}Last Linear layer name:", last_fc_name, utils.bcolors.ENDC)

    if last_fc_name is None:
        raise ValueError(f"{utils.bcolors.FAIL}[ERROR]{utils.bcolors.ENDC} No Linear layer found in the model.")

    num_classes = last_fc_layer.out_features

    bias = last_fc_layer.bias is not None

    new_last_fc_layer = torch.nn.Linear(
        in_features=last_fc_layer.in_features,
        out_features=num_classes + 1,
        bias=bias,
        device=last_fc_layer.weight.device,
        dtype=last_fc_layer.weight.dtype,
    )

    with torch.no_grad():
        new_last_fc_layer.weight[:-1] = last_fc_layer.weight
        if bias:
            new_last_fc_layer.bias[:-1] = last_fc_layer.bias

    parts = last_fc_name.split(".")
    current_module = model
    for part in parts[:-1]:
        current_module = getattr(current_module, part)
    setattr(current_module, parts[-1], new_last_fc_layer)


def boundary_exp(model, datasets, run, args):
    assert args.world_size == 1, "Boundary Expansion is not compatible with distributed training"
    assert args.task == "classification", "Boundary Expansion is not compatible with multilabel classification"

    forget_dataset = datasets.get_unlearning_data(train=args.use_train_aug)["forget"]
    forget_loader = torch.utils.data.DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    utils.print_info(args, model, forget_loader)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    unlearned_model = copy.deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)

    expand_model(unlearned_model)

    optimizer = utils.get_optimizer(unlearned_model, args)
    scheduler = utils.get_scheduler(optimizer, args)

    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        unlearned_model.train()
        for image, target, sensitive in tqdm(forget_loader, leave=False, dynamic_ncols=True):
            with torch.autocast(device_type="cuda", dtype=args.dtype):
                image = image.to(device=args.device, dtype=args.dtype)
                target = target.to(args.device)
                target = torch.ones_like(target) * args.num_classes
                sensitive = sensitive.to(args.device)

                output = unlearned_model(image)
                loss = criterion(output, target)

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

        scheduler.step()

        if args.debug:
            break

    return unlearned_model
