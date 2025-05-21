import os
os.environ['CUDA_VISIBLE_DEVICES']='0' 
import sys
import time
import argparse
import logging
import numpy as np
from predataset import *
import Modules_attention
import util
import torch
from torchvision.utils import save_image
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)
best_acc = 0
best_loss = 9999999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("using {} device.".format(device))

def train_frequency_representation(args, fr_module, fr_optimizer, fr_criterion, fr_scheduler, train_loader,val_loader,
                                   xgrid, epoch,loss_func):
    """                             
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.train()
    loss_all = 0


    for batch_idx, (noisy_sig, target_img1, target_img2) in enumerate(train_loader):
        if args.use_cuda:
            noisy_sig, target_img1, target_img2, fr_module  = noisy_sig.cuda(), target_img1.cuda(), target_img2.cuda(), fr_module.cuda()
        img1, img2 = fr_module(noisy_sig)

        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)

        target_img1 = target_img1.squeeze(1)
        target_img2 = target_img2.squeeze(1)
        target_img1 = target_img1.to(torch.float32)
        target_img2 = target_img2.to(torch.float32)

 
        loss_mse1 = loss_func(img1,target_img1)
        loss_mse2 = loss_func(img2,target_img2)

        loss_all = loss_mse1 + loss_mse2
        fr_optimizer.zero_grad()
        loss_all.backward()
        fr_optimizer.step()



    fr_module.eval()

    for batch_idx, (noisy_sig, target_img1, target_img2) in enumerate(val_loader):
        if args.use_cuda:
            noisy_sig, target_img1, target_img2, fr_module  = noisy_sig.cuda(), target_img1.cuda(), target_img2.cuda(), fr_module.cuda()
        with torch.no_grad():
            img1, img2 = fr_module(noisy_sig)
            
        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)

        target_img1 = target_img1.squeeze(1)
        target_img2 = target_img2.squeeze(1)
        target_img1 = target_img1.to(torch.float32)
        target_img2 = target_img2.to(torch.float32)        

        loss_mse1_val = loss_func(img1, target_img1)
        loss_mse2_val = loss_func(img2, target_img2)


    loss_all_val = loss_mse1_val + loss_mse2_val

    global best_loss
    if loss_all_val < best_loss: 
        print("Modified model")
        best_loss = loss_all_val
        torch.save(fr_module, "./TFA-Net/Separation/separation.pth")


    fr_scheduler.step(loss_all_val)



    logger.info("Epochs: %d / %d, Time: %.1f, MSE loss %.4f",
                epoch, args.n_epochs_fr, time.time() - epoch_start_time, loss_all_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./TFA-Net/Separation', help='output directory')
    parser.add_argument('--no_cuda',default="a", action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used during training')
    parser.add_argument('--signal_dim', type=int, default=256, help='dimensionof the input signal')
    parser.add_argument('--fr_size', type=int, default=256, help='size of the frequency representation')
    parser.add_argument('--max_n_freq', type=int, default=10,
                        help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
    parser.add_argument('--min_sep', type=float, default=0.5,
                        help='minimum separation between spikes, normalized by signal_dim')
    parser.add_argument('--distance', type=str, default='normal', help='distance distribution between spikes')
    parser.add_argument('--amplitude', type=str, default='normal_floor', help='spike amplitude distribution')
    parser.add_argument('--floor_amplitude', type=float, default=0.1, help='minimum amplitude of spikes')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=-5, help='snr parameter')
    # frequency-representation (fr) module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
    parser.add_argument('--fr_n_layers', type=int, default=20, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=16, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=3, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=256, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=2,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')

    # kernel parameters used to generate the ideal frequency representation
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]')
    parser.add_argument('--triangle_slope', type=float, default=1000,
                        help='slope of the triangle kernel normalized by signal_dim')
    parser.add_argument('--gaussian_std', type=float, default=0.12,
                        help='std of the gaussian kernel normalized by signal_dim')
    # training parameters
    parser.add_argument('--n_training', type=int, default=13200, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=5280, help='# of validation data')
    parser.add_argument('--lr_fr', type=float, default=0.001,
                        help='initial learning rate for adam optimizer used for the frequency-representation module')
    parser.add_argument('--n_epochs_fr', type=int, default=100, help= 'number of epochs used to train the fr module')
    parser.add_argument('--save_epoch_freq', type=int, default=1,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    args = parser.parse_args()
    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir,'STFTpic')):
        os.makedirs(os.path.join(args.output_dir,'STFTpic'))

    if not os.path.exists(os.path.join(args.output_dir,'Featurepig')):
        os.makedirs(os.path.join(args.output_dir,'Featurepig'))   
    
    if not os.path.exists(os.path.join(args.output_dir,'Finalpic')):
        os.makedirs(os.path.join(args.output_dir,'Finalpic'))   

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    util.print_args(logger, args)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    fr_module = Modules_attention.set_layer1_module(args)
    fr_optimizer, fr_scheduler = util.set_optim(args, fr_module, 'layer1')

    fr_criterion = torch.nn.MSELoss(reduction='sum')

    start_epoch = 1

    logger.info('[Network] Number of parameters in the frequency-representation module : %.3f M' % (
                util.model_parameters(fr_module) / 1e6))
    

    loss_func = nn.MSELoss().to(device) 


    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    for epoch in range(start_epoch, args.n_epochs_fr + 1):


        if epoch < args.n_epochs_fr:
            train_frequency_representation(args=args, fr_module=fr_module, fr_optimizer=fr_optimizer, fr_criterion=fr_criterion,
                                           fr_scheduler=fr_scheduler, train_loader=traindatas, val_loader=valdatas,
                                           xgrid=xgrid, epoch=epoch,loss_func=loss_func)


