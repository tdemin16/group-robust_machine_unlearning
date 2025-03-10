import os
import yaml
from method import utils

# YAML template for experiments
#   batch_size: 32
#   epochs: 10
#   lr: 0.1
#   clip_grad: 0.0
#   momentum: 0.9
#   weight_decay: 1e-4
#   optimizer: "sgd"
#   amsgrad: False
#   scheduler: "const"
#   t_max: float("inf")
#   step_size: 5
#   milestones:
#       - 7
#   gamma: 0.1
#   warmup_epochs: 0
#   evaluate_every: 1
#   forgetting_epochs: 5
#   use_train_aug: True
#   ssd_dampening_constant: 0.1
#   ssd_selection_weighting: 0.1
#   
#   dro_lr: 0.01


class Const(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def get_const(args) -> Const:
    # load global default const
    default_path = "method/configs/config_files/fallback.yaml"
    with open(default_path, "r") as fp:
        const_dict = yaml.safe_load(fp)

    if args.const_file is None:
        # /path/to/model
        path = os.path.join("method", "configs", "config_files", args.dataset, args.model)
        if (
            args.method != "pretrain"
            and args.method != "retrain"
        ):  
            # random -> random
            # bias -> bias_splitnumber
            unlearning_str = args.unlearning_type
            if args.unlearning_type == "bias" and args.multi_group == 0:
                unlearning_str += f"_{args.split_number}"
            elif args.unlearning_type == "bias" and args.multi_group != 0:
                unlearning_str += f"_group{args.multi_group}"
            unlearning_str += f"_{args.unlearning_ratio}"

            # /path/to/model/unlearning_type
            path = os.path.join(path, unlearning_str)

        # prepend debiasing type before method name
        method_name = args.method
        if args.rob_approach != "none" and args.method != "ga":
            method_name = args.rob_approach + "_" + method_name
        if args.method == "miu" and args.ablate_rw:
            method_name = "rw_" + method_name
        path = os.path.join(path, method_name + ".yaml")
    
    else:
        path = args.const_file

    # check config file existance
    if not os.path.isfile(path):
        print(f"{utils.bcolors.WARNING}[WARNING]{utils.bcolors.ENDC} {path} not found, using default const values.")
        path = "method/configs/config_files/fallback.yaml"
    else:
        print(f"Loading {path} config file...")

    # override global const with method const
    with open(path, "r") as fp:
        method_const_dict = yaml.safe_load(fp)
    const_dict.update(method_const_dict)
    
    const_dict["retrain_name"] = f"retrain"
    if args.rob_approach != "none" or args.pretrain_name == f"{args.rob_approach}_pretrain": # or args.method == "miu":
        const_dict["retrain_name"] = f"target_reweight_{const_dict["retrain_name"]}" # f"{args.rob_approach}_{const_dict["retrain_name"]}"
    if args.method == "miu":
        const_dict["retrain_name"] = f"target_reweight_{const_dict["retrain_name"]}"

    if const_dict["t_max"] == 'float("inf")':
        const_dict["t_max"] = float("inf")

    if args.rob_approach == "dro":
        const_dict["dtype"] = "float32"
    else:
        const_dict["dtype"] = "bfloat16"

    const = Const(const_dict)
    return const
