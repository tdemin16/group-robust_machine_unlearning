import subprocess
import random
import time
from method import utils


def main():
    random.seed(time.time_ns())

    datasets = ["celeba", "fairface", "waterbird"]
    unlearning_types = ["bias"]
    fairness_values = ["dro", "target_reweight", "none"]
    pretrain_names = ["pretrain"]
    split_numbers = [0]
    unlearning_ratios = [0.1, 0.5, 0.9]
    for method in [
        "miu",
        "l1_sparse",
        "pretrain",
        "retrain",
        "salun",
        "scrub",
    ]:
        dataset = random.sample(datasets, k=1)[0]
        unlearning_type = random.sample(unlearning_types, k=1)[0]
        fairness_value = random.sample(fairness_values, k=1)[0]
        if method in ("miu", "pretrain"):
            fairness_value = "none"
        pretrain_name = random.sample(pretrain_names, k=1)[0]
        split_number = random.sample(split_numbers, k=1)[0]
        unlearning_ratio = random.sample(unlearning_ratios, k=1)[0]
        command = f"python main.py --method {method} --dataset {dataset} --unlearning_type {unlearning_type} --rob_approach {fairness_value} --pretrain_name {pretrain_name} --split_number {split_number} --unlearning_ratio {unlearning_ratio} --debug"
        print(f"{utils.bcolors.BOLD}$ {command}{utils.bcolors.ENDC}")
        subprocess.run(command, shell=True)
        print("")


if __name__ == "__main__":
    main()
