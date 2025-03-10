import numpy as np
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

from method.metrics import compute_unlearning_metrics
from method.models import get_model


class FakeScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Mimic scheduler, but does not change lr.
    Useful to keep code simple.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.lr = optimizer.param_groups[0]["lr"]

    def step(self):
        pass

    def get_last_lr(self) -> torch.List[float]:
        return [self.lr]


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class LinearScheduler:
    def __init__(self, initial_value: float = 1, final_value: float = 0, n_iterations: int = 10):
        self.initial_value = initial_value
        self.final_value = final_value
        self.n_iterations = n_iterations
        self.current_iteration = -1

    def step(self):
        assert self.current_iteration < self.n_iterations, "LinearScheduler is done"
        self.current_iteration += 1
        magnitude = (self.current_iteration) / self.n_iterations
        step = (self.initial_value - self.final_value) * magnitude
        return self.initial_value - step


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def set_params(model: torch.nn.Module, requires_grad: bool) -> None:
    """
    Set all model parameters to a specific value.
    Params:
    -----
    model: torch.nn.Moudle
        Training model
    requires_grad: bool
        requires grad value to set
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    """
    Get oprimizer for :model:.
    Arguments are retrieved from args.
    """
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad,
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, args) -> torch.optim.lr_scheduler.LRScheduler:
    if args.scheduler == "const":
        scheduler = FakeScheduler(optimizer)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    elif args.scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs
        )
    else:
        raise ValueError(f"Unknown scheduler {args.scheduler}")

    return scheduler


def get_warmup_scheduler(
    optimizer: torch.optim.Optimizer, len_dataloader: int, args
) -> torch.optim.lr_scheduler.LRScheduler:
    if args.warmup_epochs > 0:
        warmup_scheduler = WarmUpLR(optimizer, len_dataloader * args.warmup_epochs)
    else:
        warmup_scheduler = FakeScheduler(optimizer)
    return warmup_scheduler


def get_criterion(
    criterion: str,
    reduction="mean",
    dro: str = "none",
    dro_lr: float = 0.0,
    num_groups: int = 0,
    num_sensitive_attr: int = 2,
) -> torch.nn.Module | None:
    if criterion == "binary_cross_entropy":
        return torch.nn.BCEWithLogitsLoss(reduction=reduction)
    elif criterion == "cross_entropy":
        if "dro" in dro:
            if dro == "dro":
                from method.approaches.dro import DROCrossEntropyLoss
            else:
                raise NotImplementedError
            return DROCrossEntropyLoss(dro_lr, num_groups, num_sensitive_attr)
        else:
            return torch.nn.CrossEntropyLoss(reduction=reduction)
    else:
        raise ValueError(f"Unknown criterion {criterion}")


def get_sampler(dataset, shuffle, weights: list = None):
    if shuffle:
        if weights is None:
            return torch.utils.data.RandomSampler(dataset)
        else:
            assert len(dataset) == len(weights)
            # From DRO codebase: Replacement needs to be set to True, otherwise we'll run out of minority samples
            return torch.utils.data.WeightedRandomSampler(num_samples=len(dataset), weights=weights, replacement=True)
    else:
        return torch.utils.data.SequentialSampler(dataset)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    labels = []
    for image, target, sensitive in dataloader:
        image = image.to(device)
        target = target.to(device)
        sensitive = sensitive.to(device)

        pred = model(image)

        preds.append(pred.cpu())
        labels.append(target.cpu())
        if debug:
            break
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    return preds, labels


def get_retain_metrics(datasets, args):
    name = f"{args.retrain_name}_{args.unlearning_ratio}_{args.seed}"

    unlearning_type_str = args.unlearning_type
    if args.unlearning_type == "bias" and args.multi_group == 0:
        unlearning_type_str += "_" + str(args.split_number)
    elif args.unlearning_type == "bias" and args.multi_group != 0:
        unlearning_type_str += f"_group{args.multi_group}"
    retrain_path = os.path.join(args.checkpoint_dir, args.dataset, args.model, unlearning_type_str, f"{name}.pth")
    if os.path.exists(retrain_path):
        checkpoint = torch.load(retrain_path, map_location="cuda:0", weights_only=False)
        print(f"Retrain checkpoints: {retrain_path}")
    else:
        print(f"{bcolors.WARNING}[WARNING]{bcolors.ENDC} Checkpoint {retrain_path} not found!")
        return None

    block_print()
    retrain_model = get_model(
        model_name=args.model, num_classes=args.num_classes, size=args.size, pretrained=args.pretrained
    )
    retrain_model.load_state_dict(checkpoint["state_dict"])
    retrain_model = retrain_model.to(args.device)
    retrain_model.eval()

    retrain_metrics = compute_unlearning_metrics(retrain_model, datasets, None, args)
    enable_print()
    return retrain_metrics


def print_info(args, model, dataloader):
    match args.method:
        case "miu":
            print("### MIU ###")
        case "pretrain":
            print("### Pretrain ###")
        case "retrain":
            print("### Retrain ###")
        case "scrub":
            print("### SCRUB ###")
        case "bad_teacher":
            print("### BAD TEACHER ###")
        case "rl":
            print("### Random Labeling ###")
        case "salun":
            print("### SalUn ###")
        case "ssd":
            print("### Selective Synapse Dampening ###")
        case "l1_sparse":
            print("### L1 Sparse ###")
        case "ft":
            print("### Fine Tune ###")
        case "ga":
            print("### Gradient Ascent ###")
        case "iu":
            print("### Influence Unlearning ###")
        case "boundary_exp":
            print("### Boundary Expansion ###")
        case "boundary_sh":
            print("### Boundary Shrink ###")
        case _:
            print(f"### {args.method.upper()} ###")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print("Arguments: ", end="")
    args_dict = args.__dict__
    for i, key in enumerate(sorted(args_dict)):
        print(f"{key}={args_dict[key]}" + ("," if i + 1 < len(args_dict) else ""), end=" ")
    print()
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"Number of Images: {len(dataloader.dataset)}")
    print(f"Number of Batches: {len(dataloader)}")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.device = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        args.device = args.rank % torch.cuda.device_count()
        if args.world_size:
            return
    else:
        print("Not using distributed mode")
        args.rank = 0
        args.world_size = 1
        args.device = 0
        args.distributed = False
        args.gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.set_device(args.device)
        return
    args.distributed = True

    torch.cuda.set_device(args.device)
    args.dist_backend = "nccl"
    print("| distributed init (rank {})".format(args.rank), flush=True)
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

    args.gpu_name = torch.cuda.get_device_name(0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    Disable printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# Disable
def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
