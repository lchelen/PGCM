import torch
import random
import numpy as np

def init_seed(opt):

    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):

    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def init_lr_scheduler(optim, opt):

    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_rate,
                                           step_size=opt.lr_scheduler_step)


def print_model_parameters(model):
    print('*****************Model Parameter*****************')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*************************************************')