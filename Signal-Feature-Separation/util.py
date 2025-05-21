import os
import torch
import errno
import Modules_attention#,modules

def model_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, args, epoch, module_type):
    checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if not os.path.exists(os.path.join(args.output_dir, module_type)):
        os.makedirs(os.path.join(args.output_dir, module_type))
    cp = os.path.join(args.output_dir, module_type, 'last.pth')
    fn = os.path.join(args.output_dir, module_type, 'epoch_'+str(epoch)+'.pth')
    torch.save(checkpoint, fn)
    symlink_force(fn, cp)


def load(fn, module_type, device = torch.device('cuda')):
    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']
    if device == torch.device('cpu'):
        args.use_cuda = False
    if module_type == 'fr':
        model = Modules_attention.set_fr_module(args)
    elif module_type == 'skip':
        model = Modules_attention.set_skip_module(args)
    elif module_type == 'fc':
        model = Modules_attention.set_fc_module(args)
    elif module_type == 'rdn':
        model = Modules_attention.set_rdn_module(args)
    elif module_type == 'layer1':
        model = Modules_attention.set_layer1_module(args)
    elif module_type == 'deepfreqRes':
        model = Modules_attention.set_deepfreqRes_module(args)
    elif module_type == 'freq':
        model = Modules_attention.set_freq_module(args)
    else:
        raise NotImplementedError('Module type not recognized')
    model.load_state_dict(checkpoint['model'])
    optimizer, scheduler = set_optim(args, model, module_type)
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, scheduler, args, checkpoint['epoch']


def set_optim(args, module, module_type):
    if module_type == 'fr':
        # params = list(module.parameters()) + list(adaptive.parameters())
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fr)
    elif module_type == 'fc':
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fc)
    elif module_type == 'skip':
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)
        # optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fr)
    elif module_type == 'rdn':
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)
        # optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fr)
    elif module_type == 'freq':
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)
        # optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fr)
    elif module_type == 'deepfreqRes':
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)
    elif module_type == 'layer1':
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)

    else:
        raise(ValueError('Expected module_type to be fr or fc but got {}'.format(module_type)))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,5)
    return optimizer, scheduler


def print_args(logger, args):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    logger.info(message)

    args_path = os.path.join(args.output_dir, 'run.args')
    with open(args_path, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
