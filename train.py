import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from horizonnet_master.dataset import PanoCorBonDataset
from horizonnet_master.misc.utils import  save_model
from horizonnet_master.inference import inference
from horizonnet_master.eval_general import test_general
from horizonnet_master.model import HorizonNet as HorizonNetv1
from model import HorizonNet_v2 as HorizonNetv2

device = torch.device('cuda')
ckpt_path='./ckpt/'
log_path='./logs/'

Architectures=['HorizonNetv2', 'HorizonNetv1']

def getData ( root_dir ,val, return_cor, archi='HorizonNetv2' ):
    '''
    Get DataLoader of data.
    '''
    dataset = PanoCorBonDataset( root_dir=root_dir , return_cor=return_cor, val=val, use_line=(archi == 'HorizonNetv2'))
    loader = DataLoader(dataset, args.batch_size_train,shuffle=True, drop_last=True,num_workers=20,worker_init_fn=lambda x: np.random.seed())
    return dataset , loader

def get_parameter_number(net):
    '''
    Count the number of parameters as well as trainable parameters in the network
    '''
    total_num=sum(p.numel() for p in net.parameters())
    trainable_num=sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total':total_num, 'Trainable':trainable_num}

def train(net, loader_train, tb_writer, archi='HorizonNetv2', loss=F.mse_loss()):
    '''
    Train process, train the model according to the given network, loss function
    '''
    for batch, data in enumerate(loader_train,0):
        print('Training->batch:{}'.format(batch))
        # Line here is the Manhatten line image
        data = data.to(device)
        losses = {}
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if archi == 'HorizonNetv2':
            y_bon_, y_cor_ = net(data[0], data[1])  # predicted values
        else:
            y_bon_, y_cor_ = net(data[0])  # predicted values
        losses['bon'] = loss(y_bon_, data[-2])
        losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, data[-1])
        losses['total'] = losses['bon'] + losses['cor']
        # Log
        for k, v in losses.items():
            k = 'train/%s' % k
            tb_writer.add_scalar(k, v.item(), args.cur_iter)
        tb_writer.add_scalar('train/lr', args.running_lr, args.cur_iter)
        loss = losses['total']
        print('Loss={}'.format(loss.item()))
        # backprop
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3.0, norm_type='inf')
        optimizer.step()
    return net

def valid(net , epoch ,tb_writer,  valid_data , archi='HorizonNetv2' , loss=F.mse_loss()):
    '''
    Validation process, compute the qualitative results and save them and the model using the method provide in HorizonNet.
    '''
    net.eval()
    valid_loss = {}
    for i in range(len(valid_data)):

        with torch.no_grad():
            if archi == 'HorizonNetv2':
                (x, line, y_bon, y_cor, gt_cor_id), img_path = valid_data[i]
                line = line[None]
            else:
                (x, y_bon, y_cor, gt_cor_id), img_path = valid_data[i]
            x, y_bon, y_cor = x[None], y_bon[None], y_cor[None]
            data = valid_data[i].to(device)
            losses = {}
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            data[0]=data[0][None]
            if archi == 'HorizonNetv2':
                data[1]=data[1][None]
                y_bon_, y_cor_ = net(data[0], data[1])  # predicted values
            else:
                y_bon_, y_cor_ = net(data[0])  # predicted values
            losses['bon'] = loss(y_bon_, data[-2][None])
            losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, data[-1][None])
            losses['total'] = losses['bon'] + losses['cor']
            # True eval result instead of training objective
            true_eval = dict([
                (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
                for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
            ])
            try:
                dt_cor_id = inference(net, data[0], None if archi != 'HorizonNetv2' else data[1], device, force_cuboid=False)[0]
                dt_cor_id[:, 0] *= 1024
                dt_cor_id[:, 1] *= 512
            except:
                dt_cor_id = np.array([
                    [k // 2 * 1024, 256 - ((k % 2) * 2 - 1) * 120]
                    for k in range(8)
                ])
            test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)
            # Record qualitative results
            losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
            losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
            losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
            losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']])

        for k, v in losses.items():
            valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)

    for k, v in valid_loss.items():
        k = 'valid/%s' % k
        tb_writer.add_scalar(k, v / len(valid_data), epoch)

    now_valid_score = valid_loss['3DIoU'] / len(valid_data)
    print('Score:{}'.format (now_valid_score))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', required=True,
                        help='experiment id to name checkpoints and logs')
    parser.add_argument('--pth', default=None,
                        help='path to load saved checkpoint.')
    parser.add_argument('--train_root_dir', default='data/st3d_train_full_raw_light/',
                        help='root directory to training dataset. ')
    parser.add_argument('--valid_root_dir', default='data/st3d_valid_full_raw_light/',
                        help='root directory to validation dataset. ')
    parser.add_argument('--batch_size_train', default=12, type=int,
                        help='training mini-batch size')
    parser.add_argument('--epochs', default=74, type=int,
                        help='epochs to train')
    parser.add_argument('--disp_iter', type=int, default=1,
                        help='iterations frequency to display')
    args = parser.parse_args()

    for ar in Architectures:
        print('Achitecture:{}'.format(ar))
        os.makedirs(os.path.join(ckpt_path, args.id, ar), exist_ok=True)
        # Get validation data as well as training data
        _, loader_train =getData(args.train_root_dir, False , False , ar)
        data_vali, _ = getData(args.train_root_dir, True , True , ar)
        # Load checkpoint
        if args.pth is not None:
            state_dict = torch.load(args.pth)
            if ar == 'HorizonNetv1' :
                net = HorizonNetv1(**state_dict['kwargs'], parallel='no')
            else :
                net = HorizonNetv2().to(device)
            net.load_state_dict(state_dict['state_dict'])
        else:
            if ar == 'HorizonNetv1':
                net = HorizonNetv1('resnet34', True, parallel='yes').to(device)
            else:
                net = HorizonNetv2().to(device)

        # Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, betas=(0.9, 0.999))

        print(get_parameter_number(net))


        print('Use L2 loss only:')
        tb_path = os.path.join(log_path, args.id, ar, 'L2')
        os.makedirs(tb_path, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_path)
        args.ckpt=os.path.join(ckpt_path, args.id, ar, 'L2')
        for i in range(1, args. epochs + 1):
            print('Epoch:{}'.format(i))
            net_L2 = train(net, loader_train, tb_writer , ar, F.mse_loss())
            valid( net_L2 , i , tb_writer , data_vali , ar , F.mse_loss())

        print('Use L2 + L1 loss:')
        tb_path = os.path.join(log_path, args.id, ar, 'L2+L1')
        os.makedirs(tb_path, exist_ok=True)
        tb_writer_L2_L1 = SummaryWriter(log_dir=tb_path)
        args.ckpt = os.path.join(ckpt_path , args.id , 'L2+L1')
        for i in range(1, args.epochs + 1):
            print('Epoch:{}'.format(i))
            net_L2_L1 = train(net , loader_train, tb_writer_L2_L1 , ar, F.mse_loss())
            if i >= args.epochs//2:
                valid(net_L2_L1 , i , tb_writer_L2_L1 , data_vali , ar , F.l1_loss())
            else:
                valid(net_L2_L1 , i , tb_writer_L2_L1 , data_vali , ar , F.l1_loss())
            # Save the model as .pth file every epoch.
            save_model(net,os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % i), args)












