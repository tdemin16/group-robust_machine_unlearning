import torch
from method import utils

def compute_weights(datasets, split: str):
    """
    Compute the sampling frequency for each datapoint
    """
    if split == "train":
        print(
            f"{utils.bcolors.WARNING}[WARNING]{utils.bcolors.ENDC} target dro or target reweight is used using train split as target. This does not induce any behavioural change from standard ERM training"
        )

    # get target and sensitive attributes, and number of groups
    train_target_attr, train_sensitive_attr, train_num_groups = datasets.get_attrs(split="train")
    retain_target_attr, retain_sensitive_attr, retain_num_groups = datasets.get_attrs(split=split)

    # compute number of sensitive attributs (necessary to compute group assignment)
    num_sensitive = len(set(train_sensitive_attr))

    # convert to torch tensor
    train_target_attr = torch.tensor(train_target_attr)
    train_sensitive_attr = torch.tensor(train_sensitive_attr)
    retain_target_attr = torch.tensor(retain_target_attr)
    retain_sensitive_attr = torch.tensor(retain_sensitive_attr)

    # compute group assignment for each sample in train and retain sets
    train_group = train_sensitive_attr + train_target_attr * num_sensitive
    retain_group = retain_sensitive_attr + retain_target_attr * num_sensitive

    # compute group counts
    train_group_counts = [0] * train_num_groups
    retain_group_counts = [0] * retain_num_groups
    for g in train_group:
        train_group_counts[g] += 1
    for g in retain_group:
        retain_group_counts[g] += 1
    train_group_counts = torch.tensor(train_group_counts)
    retain_group_counts = torch.tensor(retain_group_counts)

    # compute weight of sampling from each group (1 if group freq is unaltered, >1 if group freq is reduced)
    group_weights = train_group_counts / retain_group_counts
    print("Group Weights", group_weights)

    # assign weight to each sample based on group
    weights = group_weights[retain_group].tolist()
    return weights, retain_group_counts, num_sensitive, train_num_groups
    