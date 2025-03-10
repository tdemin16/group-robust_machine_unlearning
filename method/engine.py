import copy
import math
import torch
from torch.nn import functional as F
from typing import Iterable
from tqdm import tqdm

from method.metrics import mean_average_precision


def train(
    model: torch.nn.Module,
    train_dataloader: Iterable,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: float,
    warmup_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    warmup_epochs: int,
    device: torch.device,
    debug: bool,
    task: str,
    args,
):

    model.train()
    # store epoch losses and accuracies
    losses = []
    accuracies = []

    # init progress bar
    pbar = tqdm(train_dataloader, total=len(train_dataloader), leave=False, dynamic_ncols=True)
    for image, target, sensitive in pbar:
        with torch.autocast(device_type="cuda", dtype=args.dtype):
            image = image.to(device=device, dtype=args.dtype)
            target = target.to(device)
            sensitive = sensitive.to(args.device)
            output = model(image)

            # use bce_with_logits_loss (requires float target)
            if task == "multilabel":
                loss = criterion(output, target.float())
            # basic CE
            elif task == "classification":
                if "dro" not in args.rob_approach:
                    loss = criterion(output, target)
                else:
                    loss = criterion(output, target, sensitive)
            else:
                raise ValueError(f"Unknown task {task}")

        losses.append(loss.item())

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            exit()

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()

        # compute mAP for multilabel
        if task == "multilabel":
            acc = mean_average_precision(output, target, task) * 100
        # compute accuracy for classification
        elif task == "classification":
            acc = (output.argmax(dim=-1) == target).float().mean().item() * 100
        else:
            raise ValueError(f"Unknown criterion {criterion}")
        accuracies.append(acc)

        # metric label
        metric = "mAP" if task == "multilabel" else "Acc"

        # print mean loss and acc so far (in future replace with ema)
        train_loss = sum(losses) / len(losses)
        train_acc = sum(accuracies) / len(accuracies)

        # update tqdm description
        pbar_description = (
            f'| Lr: {optimizer.param_groups[0]["lr"]:.4f} | Loss: {train_loss:.4f} | {metric}: {train_acc:.2f} |'
        )
        pbar.set_description(pbar_description)

        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()

        if debug:
            break

    # return training stats
    train_stats = {"loss": train_loss, "acc": train_acc}

    return train_stats


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_dataloader: Iterable,
    criterion: torch.nn.Module,
    device: torch.device,
    debug: bool,
    task: str,
    args,
):
    model.eval()
    preds = []
    targets = []
    full_losses = []

    for image, target, sensitive in tqdm(test_dataloader, leave=False, dynamic_ncols=True):
        with torch.autocast(device_type="cuda", dtype=args.dtype):
            image = image.to(device=device, dtype=args.dtype)
            target = target.to(device)
            output = model(image)

            # same as above
            if task == "multilabel":
                loss = criterion(output, target.float())
            elif task == "classification":
                # not elegant
                full_loss = F.cross_entropy(output, target, reduction="none")
            else:
                raise ValueError(f"Unknown task {task}")

        preds.append(output.cpu())
        targets.append(target.cpu())
        full_losses.append(full_loss.cpu())

        if debug:
            break

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    full_losses = torch.cat(full_losses, dim=0)

    if task == "multilabel":
        test_acc = mean_average_precision(preds, targets, task) * 100
    elif task == "classification":
        test_acc = (preds.argmax(dim=-1) == targets).float().mean().item() * 100
    else:
        raise ValueError(f"Unknown task {task}")

    return {"loss": full_losses.mean(), "acc": test_acc, "losses": full_losses}


def train_and_eval(
    model: torch.nn.Module,
    train_dataloader: Iterable,
    test_dataloader: Iterable,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grad: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    warmup_epochs: int,
    device: torch.device,
    debug: bool,
    run,
    task: str,
    evaluate_every: int,
    args,
):

    best_acc = 0.0
    best_model = copy.deepcopy(model)

    num_digits = len(str(epochs))
    metric = "mAP" if task == "multilabel" else "Acc"

    for epoch in range(epochs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        train_stats = train(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            clip_grad=clip_grad,
            warmup_scheduler=warmup_scheduler,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            device=device,
            debug=debug,
            task=task,
            args=args,
        )
        end.record()
        torch.cuda.synchronize()

        if epoch % evaluate_every == 0:
            val_stats = evaluate(
                model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                device=device,
                debug=debug,
                task=task,
                args=args,
            )

        curr_lr = optimizer.param_groups[0]["lr"]

        log_string = f"| Epoch {str(epoch + 1).zfill(num_digits)}/{epochs} | Lr: {curr_lr:.4f} | Train Loss: {train_stats['loss']:.4f} | Train {metric}: {train_stats['acc']:.2f} |"
        if epoch % evaluate_every == 0:
            log_string += f" Val Loss: {val_stats['loss']:.4f} | Val {metric}: {val_stats['acc']:.2f} |"
        log_string += f" Time: {start.elapsed_time(end) / 1000:.1f} |"

        print(log_string)

        if run is not None:
            log_dict = {
                "train/lr": curr_lr,
                "train/loss": train_stats["loss"],
                f"train/{metric}": train_stats["acc"],
            }
            if epoch % evaluate_every == 0:
                log_dict["val/loss"] = val_stats["loss"]
                log_dict[f"val/{metric}"] = val_stats["acc"]
            run.log(log_dict)

        if epoch >= warmup_epochs:
            scheduler.step()

        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            best_model = copy.deepcopy(model)

    print(f"| Best Val {metric}: {best_acc:.2f} | Last Val {metric}: {val_stats['acc']:.2f} |")
    return best_model, model
