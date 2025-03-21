############################################################################
# SSD: Selective Synapse Dampening
# Code adapted from https://github.com/if-loops/selective-synaptic-dampening
############################################################################

import torch
from functools import partial
from typing import Dict, List

from method import utils


class SelectiveSynapseDampening:
    def __init__(
        self,
        model,
        criterion,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict([(k, torch.zeros_like(p, device=p.device)) for k, p in model.named_parameters()])

    def calc_importance(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            x, y = batch
            self.opt.zero_grad()
            x, y = x.to(device), y.to(device)
            out = self.model(x)

            # retain identities and use them with cross entropy loss
            if isinstance(out, tuple):
                assert isinstance(self.criterion, torch.nn.CrossEntropyLoss)
                out = out[1]

            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                y = y.float()
            loss = self.criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(self.model.named_parameters(), importances.items()):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        num_locations = 0
        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)
                num_locations += len(locations[0])

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(self.exponent)
                update = weight[locations]
                
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                num_locations -= len(min_locs[0])
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)

        print(f"Number of locations updated: {num_locations}")


def ssd(model, datasets, run, args):
    assert args.world_size == 1, "SSD is not compatible with distributed training"
    support_dataset = datasets.get_unlearning_data(train=False)["forget"]
    train_dataset = datasets.get_train_data(train=False)

    partial_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    support_loader = partial_loader(support_dataset)
    train_loader = partial_loader(train_dataset)

    criterion = utils.get_criterion(args.criterion).to(args.device)
    optimizer = utils.get_optimizer(model, args)

    utils.print_info(args, model, support_loader)

    model.eval()

    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "dampening_constant": args.ssd_dampening_constant,
        "selection_weighting": args.ssd_selection_weighting,
    }

    ssd = SelectiveSynapseDampening(model, criterion, optimizer, args.device, parameters)
    support_importance = ssd.calc_importance(support_loader, args.device)
    train_importance = ssd.calc_importance(train_loader, args.device)
    ssd.modify_weight(train_importance, support_importance)

    return model
