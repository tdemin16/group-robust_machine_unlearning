import copy
import torch
from functools import partial
from tqdm import tqdm
from torch.autograd import grad

from method import utils


def sample_grad(model, loss):
    params = [param for param in model.parameters() if param.requires_grad]
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


def compute_grad(model, loader, criterion, args):
    params = [param.view(-1) for param in model.parameters() if param.requires_grad]
    grad_ = torch.zeros_like(torch.cat(params), device=args.device)
    
    total = 0
    model.eval()

    for i, (image, target, sensitive) in enumerate(loader):
        batch_size = image.size(0)
        with torch.autocast(device_type="cuda", dtype=args.dtype):
            image = image.to(device=args.device, dtype=args.dtype)
            target = target.to(args.device)

            output = model(image)
            loss = criterion(output, target)
        grad_ += sample_grad(model, loss) * batch_size
        total += batch_size
        
        if args.debug:
            break

    return grad_, total


def woodfisher(model, loader, criterion, v, args):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000 if args.size == 32 else 300000
    o_vec = None
    for idx, (data, label) in enumerate(tqdm(loader, desc="Woodfisher", leave=False, dynamic_ncols=True)):
        model.zero_grad()
        data = data.to(args.device)
        label = label.to(args.device)
        output = model(data)

        loss = criterion(output, label)
        s_grad = sample_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(s_grad)
            else:
                tmp = torch.dot(o_vec, s_grad)
                k_vec -= (torch.dot(k_vec, s_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec

        if args.debug:
            break
    return k_vec


def apply_perturb(model, v):
    curr = 0
    for param in model.parameters():
        if param.requires_grad:
            length = param.view(-1).shape[0]
            param.view(-1).data += v[curr : curr + length].data
            curr += length


def iu(model, datasets, run, args):
    assert args.world_size == 1, "IU is not compatible with distributed training"
    assert args.task == "classification", "IU is not compatible with multilabel classification"

    if args.dro:
        print(f"{utils.bcolors.WARNING}[WARNING]{utils.bcolors.ENDC} DRO flag is set to true but Influence Unlearning does not compute gradient descent on retain data. DRO flag will be ignored...")

    train_dataset = datasets.get_train_data()
    unlearning_datasets = datasets.get_unlearning_data(train=args.use_train_aug)
    retain_dataset = unlearning_datasets["retain"]
    forget_dataset = unlearning_datasets["forget"]
    
    generic_loader = partial(
        torch.utils.data.DataLoader,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )
    train_loader = generic_loader(train_dataset)
    retain_loader = generic_loader(retain_dataset)
    retain_grad_loader = generic_loader(retain_dataset, batch_size=1)
    forget_loader = generic_loader(forget_dataset)

    utils.print_info(args, model, train_loader)

    unlearned_model = copy.deepcopy(model)
    unlearned_model.to(args.device)

    criterion = utils.get_criterion(args.criterion).to(args.device)

    f_grad, f_total = compute_grad(unlearned_model, forget_loader, criterion, args)
    r_grad, r_total = compute_grad(unlearned_model, retain_grad_loader, criterion, args)

    r_grad *= f_total / ((f_total + r_total) * r_total)
    f_grad /= f_total + r_total

    perturb = woodfisher(unlearned_model, retain_loader, criterion, f_grad - r_grad, args)
    apply_perturb(unlearned_model, args.alpha_hessian * perturb)

    return unlearned_model
