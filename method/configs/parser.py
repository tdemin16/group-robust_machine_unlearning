import argparse
import os
import torch
from method import utils
from method.configs.const import get_const


def str2bool(v):
    """Converts a string to a boolean value."""
    return v.lower() in ("true", "t", "y", "yes", "1")


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # --- Logging info --- #
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--description", type=str, default="")

    # --- Model Hyperparameters --- #
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=[
            "vit_base_patch16_224",
            "resnet18",
        ],
    )
    parser.add_argument("--pretrained", type=str2bool, default="True")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--pretrain_name", type=str, default="pretrain")

    # --- Dataset --- #
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celeba", "fairface", "waterbird"],
        default="fairface",
    )
    parser.add_argument("--data_dir", type=str, default="/home/tdemin/datasets/")
    parser.add_argument(
        "--download", type=str2bool, default="False", help="Download the dataset when possible."
    )

    # --- Task Hyperparameters --- #
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "miu",
            "bad_teacher",
            "boundary_exp",
            "boundary_sh",
            "ft",
            "ga",
            "iu",
            "l1_sparse",
            "pretrain",
            "retrain",
            "rl",
            "salun",
            "scrub",
            "ssd",
        ],
        default="pretrain",
    )
    parser.add_argument(
        "--rob_approach",
        type=str,
        default="none",
        choices=["none", "dro", "subsample", "target_reweight"],
        help="Which approach for group robustness to use. none: no robust optimization; dro: use group DRO; subsample: subsample dataset based on grup information; target_reweight: reweight sampling targeting original data distribution.",
    )
    parser.add_argument("--unlearning_type", type=str, default="bias", choices=["bias", "random"])
    parser.add_argument(
        "--unlearning_ratio", type=float, default=0.5, help="Ratio of data to unlearn"
    )
    parser.add_argument(
        "--split_number",
        type=int,
        default=0,
        help="Which unlearning split to use in bias unlearning.",
    )
    parser.add_argument(
        "--multi_group", type=int, default=0, help="Unlearn multiple groups rather than one."
    )

    # --- Ablations --- #
    # before getting const to get correct yaml
    parser.add_argument("--ablate_rw", action="store_true")
    parser.add_argument("--ablate_alignment", action="store_true")
    parser.add_argument("--ablate_unlearn", action="store_true")
    parser.add_argument("--ablate_retain", action="store_true")

    # --- Custom const yaml --- #
    parser.add_argument("--const_file", type=str, default=None)

    # --- Get Constants --- #
    tmp_args, _ = parser.parse_known_args()
    const = get_const(tmp_args)

    parser.add_argument("--retrain_name", type=str, default=const.retrain_name)

    # --- Our Hyperparameters --- #
    parser.add_argument("--control_bias", type=str2bool, default=True)
    parser.add_argument("--mine_lr", type=float, default=const.mine_lr)
    parser.add_argument("--lambda_mi", type=float, default=const.lambda_mi)

    # --- L1 Sparse Hyperparameters --- #
    parser.add_argument("--l1_penalty", type=float, default=5e-4)

    # --- IU Hyperparameters --- #
    parser.add_argument("--alpha_hessian", type=float, default=5)

    # --- SCRUB Hyperparameters --- #
    parser.add_argument("--alpha_scrub", type=float, default=0.001)
    parser.add_argument("--gamma_scrub", type=float, default=0.99)
    parser.add_argument("--forgetting_epochs", type=int, default=const.forgetting_epochs)
    parser.add_argument("--temperature_scrub", type=float, default=4)
    parser.add_argument("--batch_size_retain", type=int, default=128)
    parser.add_argument("--use_train_aug", type=str2bool, default=const.use_train_aug)

    # --- SalUn Hyperparameters --- #
    parser.add_argument("--saliency_threshold", type=float, default=0.5)

    # --- SSD Hyperparameters --- #
    parser.add_argument(
        "--ssd_dampening_constant", type=float, default=const.ssd_dampening_constant
    )
    parser.add_argument(
        "--ssd_selection_weighting", type=float, default=const.ssd_selection_weighting
    )

    # --- DRO Hyperparameters --- #
    parser.add_argument("--dro_lr", type=float, default=const.dro_lr)
    parser.add_argument("--compute_freq", type=str2bool, default=const.compute_freq)
    parser.add_argument("--train_aug_group", type=str2bool, default=True)

    # --- Training Hyperparameters --- #
    parser.add_argument("--batch_size", type=int, default=const.batch_size)
    parser.add_argument("--epochs", type=int, default=const.epochs)
    parser.add_argument("--lr", type=float, default=const.lr)
    parser.add_argument("--clip_grad", type=float, default=const.clip_grad)
    parser.add_argument("--momentum", type=float, default=const.momentum)
    parser.add_argument("--weight_decay", type=float, default=const.weight_decay)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam", "adamw"],
        default=const.optimizer,
    )
    parser.add_argument(
        "--amsgrad",
        type=str2bool,
        default=const.amsgrad,
        help="Whether to use the AMSGrad variant of Adam",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["const", "step", "multistep", "cosine", "exp", "linear"],
        default=const.scheduler,
    )
    parser.add_argument(
        "--t_max",
        type=int,
        default=const.t_max,
        help="Number of epochs for cosine annealing scheduler",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=const.step_size,
        help="Number of epochs for step scheduler",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=const.milestones,
        help="Milestones for multistep scheduler",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=const.gamma,
        help="Gamma for step and multistep scheduler",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=const.warmup_epochs,
        help="Number of epochs for warmup scheduler",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=const.evaluate_every,
        help="Number of epochs to evaluate the model",
    )
    parser.add_argument("--dtype", type=str, default=const.dtype)

    # --- DataLoader Hyperparameters --- #
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--drop_last", type=str2bool, default="False")
    parser.add_argument("--pin_memory", type=str2bool, default="True")

    # --- Misc --- #
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disables saving of checkpoints, logging, and set epochs to 1",
    )
    parser.add_argument("--save", type=str2bool, default="True")
    parser.add_argument("--store_dir", type=str, default="output/")

    # --- Logging --- #
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="unlearning")
    parser.add_argument("--entity", type=str, default="tdemin")
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--store_csv", action="store_true")
    parser.add_argument("--csv_store_dir", type=str, default="experiments/")

    parser.add_argument("--store_curves", action="store_true")
    parser.add_argument("--curves_store_dir", type=str, default="curves/")

    return parser


def custom_bool(v):
    """Converts a string to a boolean value."""
    return v.lower() in ("true", "t", "y", "yes", "1")


def parse_args() -> argparse.Namespace:
    parser = get_argparser()
    args = parser.parse_args()

    if args.unlearning_type == "bias":
        assert args.dataset in ("fairface", "celeba", "waterbird")
    elif args.unlearning_type == "random":
        assert args.dataset in ("fairface", "celeba", "waterbird")
    else:
        raise ValueError(f"Unknown unlearning type {args.unlearning_type}")

    if args.dataset in ("celeba", "fairface", "waterbird"):
        args.criterion = "cross_entropy"
        args.task = "classification"

    if args.multi_group:
        assert args.dataset == "fairface", "Multi-group unlearning is available only for fairface"
        assert args.multi_group > 1, "Multi-group requires at least 2 groups to be unlearned"
        assert args.split_number == 0, "Either multi_group or split_number can be different than 0"

    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    args.store_dir = os.path.abspath(args.store_dir)

    if args.t_max > args.epochs:
        args.t_max = args.epochs - args.warmup_epochs

    if args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    elif args.dtype == "float32":
        args.dtype = torch.float32
    else:
        raise NotImplementedError

    if args.debug:
        args.epochs = 1
        args.name = "debug"
        args.save = False
        args.project = "debug"

    return args
