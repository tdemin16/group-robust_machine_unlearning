import copy
import torch
from tqdm import tqdm

from method import utils
from method.metrics import evaluate_after_unlearning


def FGSM_perturb(model, image, target, bound, criterion):
    model.eval()
    image_adv = image.detach().clone().requires_grad_(True)

    output = model(image_adv)
    loss = criterion(output, target)
    loss.backward()

    grad_sign = image_adv.grad.data.detach().sign()
    image_adv = image_adv + grad_sign * bound
    image_adv = torch.clamp(image_adv, 0.0, 1.0)
    image_adv = torch.round(image_adv * 255) / 255

    model.zero_grad()
    return image_adv.detach()


def boundary_sh(model, datasets, run, args):
    assert args.world_size == 1, "Boundary Shrink is not compatible with distributed training"
    assert args.task == "classification", "Boundary Shrink is not compatible with multilabel classification"

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

    optimizer = utils.get_optimizer(unlearned_model, args)
    scheduler = utils.get_scheduler(optimizer, args)

    num_digits = len(str(args.epochs))
    for epoch in range(args.epochs):
        unlearned_model.train()
        for image, target, sensitive in tqdm(forget_loader, leave=False, dynamic_ncols=True):
            with torch.autocast(device_type="cuda", dtype=args.dtype):
                image = image.to(device=args.device, dtype=args.dtype)
                target = target.to(args.device)
                sensitive = sensitive.to(args.device)

                image_adv = FGSM_perturb(model, image, target, 0.1, criterion)
                with torch.no_grad():
                    adv_outputs = model(image_adv)
                adv_target = torch.argmax(adv_outputs, dim=1)

                output = unlearned_model(image)
                loss = criterion(output, adv_target)

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
