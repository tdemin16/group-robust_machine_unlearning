import torch
import torchmetrics
import warnings

from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from method import utils
from method.dataset.dataset_classes import UnlearningDataset


def mean_average_precision(logits: torch.Tensor, targets: torch.Tensor, task: str):
    with warnings.catch_warnings(action="ignore"):
        logits = torch.nn.functional.sigmoid(logits)
        return torchmetrics.functional.average_precision(logits, targets, task=task, num_labels=logits.size(1))


def evaluate_after_unlearning(
    model: torch.nn.Module,
    datasets: UnlearningDataset,
    criterion: torch.nn.Module,
    args=None,
):
    from method.engine import evaluate  # avoid circular import

    # get retain, forget, and test data
    unlearning_data = datasets.get_unlearning_data(train=False)
    unlearning_data.update({"val": datasets.get_val_data()})
    unlearning_data.update({"test": datasets.get_test_data()})

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    retain_loader = partial_loader(unlearning_data["retain"])
    forget_loader = partial_loader(unlearning_data["forget"])
    val_loader = partial_loader(unlearning_data["val"])
    test_loader = partial_loader(unlearning_data["test"])

    # evaluate model on retain, forget, and test set
    retain_stats = evaluate(model, retain_loader, criterion, args.device, args.debug, args.task, args)
    forget_stats = evaluate(model, forget_loader, criterion, args.device, args.debug, args.task, args)
    val_stats = evaluate(model, val_loader, criterion, args.device, args.debug, args.task, args)
    test_stats = evaluate(model, test_loader, criterion, args.device, args.debug, args.task, args)

    return (
        retain_stats["loss"],
        retain_stats["acc"],
        retain_stats["losses"],
        forget_stats["loss"],
        forget_stats["acc"],
        forget_stats["losses"],
        val_stats["loss"],
        val_stats["acc"],
        val_stats["losses"],
        test_stats["loss"],
        test_stats["acc"],
        test_stats["losses"],
    )


@torch.no_grad()
def compute_loss(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, args):
    model.eval()
    losses = []
    for image, target, sensitive in dataloader:
        with torch.autocast(device_type="cuda", dtype=args.dtype):
            image = image.to(device=args.device, dtype=args.dtype)
            target = target.to(args.device)
            sensitive = sensitive.to(args.device)
            output = model(image)
            loss = torch.nn.functional.cross_entropy(output, target, reduction="none")

        losses.append(loss)

        if args.debug:
            break

    return torch.cat(losses, dim=0)


def compute_basic_mia(retain_losses, forget_losses, val_losses, test_losses):
    train_loss = torch.cat((retain_losses, val_losses), dim=0).unsqueeze(1).cpu().numpy()
    train_target = torch.cat((torch.ones(retain_losses.size(0)), torch.zeros(val_losses.size(0))), dim=0).numpy()
    test_loss = torch.cat((forget_losses, test_losses), dim=0).unsqueeze(1).cpu().numpy()
    test_target = torch.cat((torch.ones(forget_losses.size(0)), torch.zeros(test_losses.size(0)))).cpu().numpy()
    
    best_auc = 0
    best_acc = 0
    for n_est in [20, 50, 100]:
        for criterion in ['gini', 'entropy']:
            mia_model = RandomForestClassifier(n_estimators=n_est, criterion=criterion, n_jobs=8, random_state=0)
            mia_model.fit(train_loss, train_target)
            
            y_hat = mia_model.predict_proba(test_loss)[:, 1]
            auc = roc_auc_score(test_target, y_hat) * 100

            y_hat = mia_model.predict(forget_losses.unsqueeze(1).cpu().numpy()).mean()
            acc = (1 - y_hat) * 100

            if acc > best_acc:
                best_acc = acc
                best_auc = auc

    return best_auc, best_acc


@torch.no_grad()
def compute_equalized_odds(model, datasets, args):
    model.eval()
    # return indices of test samples to retrieve sensitive attributes
    test_data = datasets.get_test_data()
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )

    #        SA = 0  SA = 1
    # TA = 0   []      []
    # TA = 1   []      []
    accs = [[[], []], [[], []]]

    target_attr_idx, sensitive_attr_idx = datasets.get_bias_indexes()
    target_attr_idx = torch.tensor(target_attr_idx, device=args.device)
    sensitive_attr_idx = torch.tensor(sensitive_attr_idx, device=args.device)

    target_acc = []

    for image, target, sensitive in test_loader:
        image = image.to(args.device)
        target = target.to(args.device)
        sensitive = sensitive.to(args.device)
        output = model(image)
        pred = torch.argmax(output, dim=-1)

        # sa = sensitive
        # ta = target
        ta = torch.isin(target, target_attr_idx).int()
        sa = torch.isin(sensitive, sensitive_attr_idx).int()
        for sa_, ta_, p, o in zip(sa, ta, pred, target):
            accs[ta_][sa_].append((p == o).float().item())

        for p, t, s in zip(pred, target, sensitive):
            if torch.isin(t, target_attr_idx) and torch.isin(s, sensitive_attr_idx):
                target_acc.append((p==t).float().item())

    for i in range(len(accs)):
        for j in range(len(accs[i])):
            accs[i][j] = sum(accs[i][j]) / len(accs[i][j])
    
    eo = 0
    eo_size = 0
    for ta in range(len(accs)):
        for sa in range(len(accs[ta])):
            for sa_ in range(sa + 1, len(accs[ta])):
                eo_size += 1
                eo += abs(accs[ta][sa] - accs[ta][sa_])
    eo /= eo_size

    target_acc = sum(target_acc) / len(target_acc)

    return eo * 100, target_acc * 100


def compute_tow(appx, ref):
    return 1 - abs(appx - ref) / 100


def compute_gap(metrics, target):
    gap = []
    for metric in ["retain/acc", "forget/acc", "test/acc", "mia/acc", "eo", "target_acc"]:
        gap.append(abs(metrics[metric] - target[metric])) 
    return sum(gap) / len(gap)


def compute_unlearning_metrics(
    model: torch.nn.Module, datasets, run, args, retrain_metrics: dict | None = None
) -> dict:
    """
    Compute metrics for unlearning.

    TODO: It uses multiple forward passes for each dataset, each one to estimate a metric. In the future compress into one
    """
    criterion = utils.get_criterion(args.criterion).to(args.device)

    print("Evaluate after unlearning...")
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
    ) = evaluate_after_unlearning(model, datasets, criterion, args)

    print("Computing ToW...")
    tow = -1
    if retrain_metrics is not None:
        tow = (
            compute_tow(retain_acc, retrain_metrics["retain/acc"])
            * compute_tow(forget_acc, retrain_metrics["forget/acc"])
            * compute_tow(test_acc, retrain_metrics["test/acc"])
            * 100
        )
    elif args.method != "retrain":
        print(f"{utils.bcolors.WARNING}[WARNING]{utils.bcolors.ENDC} Retrain Metrics not found, skipping ToW calculation")

    print("Computing MIA...")
    mia_auc, mia_acc = compute_basic_mia(retain_losses, forget_losses, val_losses, test_losses)

    print("Computing Equalized Odds...")
    eo, target_acc = compute_equalized_odds(model, datasets, args)

    task_metrics = {
        "retain/loss": retain_loss,
        "retain/acc": retain_acc,
        "forget/loss": forget_loss,
        "forget/acc": forget_acc,
        "test/loss": test_loss,
        "test/acc": test_acc,
        "mia/auc": mia_auc,
        "mia/acc": mia_acc,
        "eo": eo,
        "target_acc": target_acc
    }
    if tow != -1:
        task_metrics["tow"] = tow

    gap = -1
    if retrain_metrics is not None:
        gap = compute_gap(task_metrics, retrain_metrics)
        task_metrics["gap"] = gap


    metric = "mAP" if args.task == "multilabel" else "Acc"
    print(
        f"| Retain Loss: {retain_loss:.4f} | Retain {metric}: {retain_acc:.2f} | Forget Loss: {forget_loss:.4f} | Forget {metric}: {forget_acc:.2f} | Test Loss: {test_loss:.4f} | Test {metric}: {test_acc:.2f} | MIA AUC: {mia_auc:.2f} | MIA Acc: {mia_acc:.2f} |",
        end="",
    )
    if tow != -1:
        print(f" ToW: {tow:.2f} |", end="")
    print(f" EO: {eo:.2f} | Target Acc: {target_acc:.2f} |", end="")
    if gap != -1:
        print(f" Avg. Gap: {gap:.2f} |", end="")
    print()

    if run is not None:
        log_dict = {
            "retain/loss": retain_loss,
            "retain/acc": retain_acc,
            "forget/loss": forget_loss,
            "forget/acc": forget_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "mia/auc": mia_auc,
            "mia/acc": mia_acc,
            "eo": eo,
            "target_acc": target_acc
        }
        if tow != -1:
            log_dict["tow"] = tow
        if gap != -1:
            log_dict["gap"] = gap

        run.log(log_dict)

    return task_metrics
