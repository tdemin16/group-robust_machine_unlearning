import csv
import os
import time
import torch
import wandb
from datetime import datetime
from method import approaches
from method import utils
from method.configs import parse_args
from method.dataset import get_datasets
from method.metrics import compute_unlearning_metrics
from method.models import get_model


def main(args):
    utils.init_distributed_mode(args)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    utils.seed_everything(args.seed)

    # setup weight and biases
    run = None
    if args.wandb and args.device == 0:
        mode = "offline" if args.offline else "online"
        run = wandb.init(
            name=args.name,
            project=args.project,
            entity=args.entity,
            config=args,
            mode=mode,
        )

    # load pretrain checkpoint
    splits = None
    state_dict = None
    unlearning_type_str = args.unlearning_type
    if args.unlearning_type == "bias" and args.multi_group == 0:
        unlearning_type_str += "_" + str(args.split_number)
    elif args.unlearning_type == "bias" and args.multi_group != 0:
        unlearning_type_str += f"_group{args.multi_group}"
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        args.dataset,
        args.model,
        unlearning_type_str,
        f"{args.pretrain_name}_{args.unlearning_ratio}_{args.seed}.pth",
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
        splits = checkpoint["splits"]
        state_dict = checkpoint["state_dict"]
    else:
        print(f"{utils.bcolors.WARNING}[WARNING]{utils.bcolors.ENDC} Checkpoint {checkpoint_path} not found")

    # load dataset using stored splits if available
    datasets = get_datasets(args, splits=splits)

    # create model and load state dict if available
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        size=args.size,
        pretrained=args.pretrained,
    )
    if state_dict is not None and args.method not in ("pretrain", "retrain"):
        print("Loading State Dict...")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(args.device)

    # get unlearning algorithm
    method_fn = getattr(approaches, args.method)

    # unlearn or pretrain
    model = method_fn(model, datasets, run, args)

    model.eval()
    utils.set_params(model, requires_grad=False)

    # load retrain model to compute ToW
    if "retrain" != args.method:
        retrain_metrics = utils.get_retain_metrics(datasets, args)
    else:
        retrain_metrics = None

    # compute unlearning metrics
    metrics = compute_unlearning_metrics(model, datasets, run, args, retrain_metrics=retrain_metrics)

    if run is not None:
        run.finish()

    if args.save and args.device == 0:
        # store paths
        unlearning_str = args.unlearning_type
        if args.unlearning_type == "bias" and args.multi_group == 0:
            unlearning_str += f"_{args.split_number}"
        elif args.unlearning_type == "bias" and args.multi_group != 0:
            unlearning_str += f"_group{args.multi_group}"

        path_dir = os.path.join(args.store_dir, args.dataset, args.model, unlearning_str)
        csv_dir = os.path.join(args.csv_store_dir, args.dataset, args.model, unlearning_str)

        # create paths if they do not exist
        if not os.path.exists(path_dir):
            try:
                os.makedirs(path_dir)
            except OSError:
                # two processes created the same directory, make sure one of them stops for a bit
                time.sleep(5)
        if not os.path.exists(csv_dir):
            try:
                os.makedirs(csv_dir)
            except OSError:
                time.sleep(5)

        # store_dir/dataset/model/method_arg1_arg2_seed.pth
        name = f"{args.method}"
        if args.rob_approach != "none":
            name = args.rob_approach + "_" + name
        if args.method == "miu" and args.ablate_rw:
            name = "rw_" + name
        if args.method == "miu" and args.ablate_alignment:
            name = "align_" + name
        if args.method == "miu" and args.ablate_unlearn:
            name = "ul_" + name
        if args.method == "miu" and args.ablate_retain:
            name = "ret_" + name
        if args.method != "pretrain" and args.method != "retrain" and args.pretrain_name != "pretrain":
            name += f"_{args.pretrain_name}"
        name += f"_{args.unlearning_ratio}"
        name += f"_{args.seed}"

        checkpoint_path = os.path.join(path_dir, f"{name}.pth")
        print(f"Saving model to {checkpoint_path}")

        # pretrain should be general, we don't want to store splits
        splits = datasets.get_splits()

        # store args for future evaluation and model
        checkpoint = {
            "args": args,
            "splits": datasets.get_splits(),
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # store experiment metrics in csv
        if args.store_csv:
            csv_name = f"{args.method}"
            if args.rob_approach != "none":
                csv_name = args.rob_approach + "_" + csv_name
            if args.method == "miu" and args.ablate_rw:
                csv_name = "rw_" + csv_name
            if args.method == "miu" and args.ablate_alignment:
                csv_name = "align_" + csv_name
            if args.method == "miu" and args.ablate_unlearn:
                csv_name = "ul_" + csv_name
            if args.method == "miu" and args.ablate_retain:
                csv_name = "ret_" + csv_name
            if args.method != "pretrain" and args.method != "retrain" and args.pretrain_name != "pretrain":
                csv_name += f"_{args.pretrain_name}"
            csv_name += f"_{args.unlearning_ratio}"
            experiment_path = os.path.join(csv_dir, f"{csv_name}.csv")
            print(f"Storing experiment results to {experiment_path}")

            with open(experiment_path, "a", newline="") as fp:
                writer = csv.writer(fp)
                row = [
                    args.seed,
                    metrics["retain/acc"],
                    metrics["forget/acc"],
                    metrics["test/acc"],
                    metrics["mia/auc"],
                    metrics["mia/acc"],
                ]
                if "tow" in metrics:
                    row.append(metrics["tow"])
                else:
                    row.append("-")
                if "eo" in metrics:
                    row.append(metrics["eo"])
                    row.append(metrics["target_acc"])
                else:
                    row.append("-")
                if "gap" in metrics:
                    row.append(metrics["gap"])
                else:
                    row.append("-")
                row.append(datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))
                writer.writerow(row)


if __name__ == "__main__":
    args = parse_args()

    main(args)
