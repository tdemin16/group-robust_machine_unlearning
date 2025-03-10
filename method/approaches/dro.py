"""
Adapted from https://github.com/kohpangwei/group_DRO
"""

import torch


def compute_weights(datasets, split: str):
    """
    Compute the sampling frequency for each datapoint
    """
    # get target and sensitive attributes, and number of groups
    target_attr, sensitive_attr, num_groups = datasets.get_attrs(split=split)

    # num of sensitiven attrs
    num_sensitive_attr = len(set(sensitive_attr))

    # convert tensor
    target_attr = torch.tensor(target_attr)
    sensitive_attr = torch.tensor(sensitive_attr)

    group = sensitive_attr + target_attr * num_sensitive_attr

    # count number of samples for each group
    group_counts = [0] * num_groups
    for g in group:
        group_counts[g] += 1
    group_counts = torch.tensor(group_counts)

    # normalize group weights
    group_weights = len(sensitive_attr) / group_counts
    print("Group Weights", group_weights)

    # assign weight to each sample based on group
    weights = group_weights[group].tolist()

    return weights, group_counts, num_sensitive_attr, num_groups


class DROCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Computes Cross-Entropy Loss as described in https://github.com/kohpangwei/group_DRO
    TL;DR: Scales the loss value of each sample by the group weight
    """

    def __init__(self, dro_lr: float, num_groups: int, num_sensitive_attr: int, *args, **kwargs):
        """
        It instantiates a distributionally robust CrossEntropyLoss with reduction == "none"

        Params
        ------
        dro_lr: float
            DRO step size

        num_groups: int
            Sensitive Attributes cardinality
        """
        kwargs["reduction"] = "none"
        super().__init__(**kwargs)

        self.dro_lr = dro_lr
        self.num_groups = num_groups
        self.num_sensitive_attr = num_sensitive_attr

        self.register_buffer("adv_probs", torch.ones(num_groups) / self.num_groups)

    @staticmethod
    def compute_group_average(loss, group, num_groups):
        group_map = (group == torch.arange(num_groups, device=loss.device).unsqueeze(1).long()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_loss = (group_map @ loss.view(-1)) / group_denom
        return group_loss

    def forward(self, input, target, sensitive):
        per_sample_loss = super().forward(input, target)
        group = sensitive + target * self.num_sensitive_attr
        group_loss = self.compute_group_average(per_sample_loss, group, self.num_groups)
        self.adv_probs = self.adv_probs * torch.exp(self.dro_lr * group_loss.data)
        self.adv_probs = self.adv_probs / self.adv_probs.sum()
        
        loss = group_loss @ self.adv_probs
        return loss
