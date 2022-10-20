# python check_OOD_DenseNet.py --cuda --gpu 1 --net ./densenet_svhn.pth --ood_dataset LSUN 

from __future__ import print_function
import argparse
import os
import random
from tkinter import N
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
from torchvision import datasets
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import math
import os 
import json
import numpy as np
import sklearn.covariance
import pickle 

from csv import writer

import PIL
from PIL import Image

import pdb

from scipy.integrate import quad_vec
from scipy import integrate

from resnet import ResNet34
from densenet import DenseNet3

# from utils import get_intermediate_features, get_intermediate_features_inference
# from utils import sample_estimator, get_Mahalanobis_stats
# from utils import get_min_max_gram, get_deviations_gram
# from utils import get_energy_stats
# from utils import get_odin_stats, get_softmax_stats
# from utils import iterate_data_gradnorm

from densenet_utils import get_min_max_gram, get_deviations_gram, get_Mahalanobis_stats, sample_estimator, get_energy_stats

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean in_dist_testue expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--resnetType', type=int, default=34, help='18/34')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path load the trained network")
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')
parser.add_argument('--shift', type=float, default=4)
parser.add_argument('--shrink', type=float, default=0.8)
parser.add_argument('--enlarge', type=float, default=1.2)

parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--trials', type=int, default=5, help='no. of trials for taking average for the final results')

# OOD detection params
parser.add_argument('--n', type=int, default=20, help='no. of transformations')
parser.add_argument('--proper_train_size', default=45000, type=int, help='proper training dataset size')
parser.add_argument('--ood_dataset', default='SVHN', help='SVHN/IMAGENET/CIFAR10/LSUN/Places365/adv_cifar10')
parser.add_argument('--energy_temperature', type=float, default = 100.0, help = 'Energy temperature')
parser.add_argument('--ODIN_temperature',type=float,  default = 100.0, help = 'ODIN temperature')
parser.add_argument('--ODIN_epsilon',type=float, default = 0.001, help = 'ODIN epsilon')

# Adv detection params
parser.add_argument('--adv_data_root', default='ResNet34_cifar10/adv_data_ResNet34_cifar10_BIM.pth', help='path to the adversarial examples')

opt = parser.parse_args()
print(opt)


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")
device = torch.device("cuda:1")

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

num_classes = 10
model_file_path = opt.net
net = DenseNet3(100, num_classes)
net.load_state_dict(torch.load(model_file_path, map_location = device))

net.to(device)
net.eval()
inputs = torch.rand((1, 3, 32, 32)).to(device)
logits, features = net.feature_list(inputs)
num_output =  len(features)

gram_tnr_list = []
gram_fpr_list = []

mahala_tnr_list = []
mahala_fpr_list = []


in_test_dataset, ood_test_dataset, cal_set, cal_dataloader, in_test_dataloader, out_test_dataloader = None, None, None, None, None, None

complete_train_dataset = datasets.SVHN(root = 'data', split='train',  download=True,
                   transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    ]))
    
in_test_dataset = datasets.SVHN(root = 'data', split='test',  download=True,
                   transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    ]))


train_dataset = torch.utils.data.Subset(complete_train_dataset, list(range(45000)))
cal_dataset = torch.utils.data.Subset(complete_train_dataset, list(range(len(complete_train_dataset) - 5000, len(complete_train_dataset))))

cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                        shuffle=False, num_workers=int(opt.workers)) 

in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                        shuffle=False, num_workers=int(opt.workers)) 

if opt.ood_dataset == 'IMAGENET':
    ood_test_dataset = datasets.ImageFolder(os.path.join('data', 'Imagenet_resize'), 
                   transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    ]))
elif opt.ood_dataset == 'LSUN':
    ood_test_dataset = datasets.ImageFolder(os.path.join('data', 'LSUN_resize'), 
                   transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    ]))
elif opt.ood_dataset == 'iSUN':
    ood_test_dataset = datasets.ImageFolder(os.path.join('data', 'iSUN'), 
                   transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    ]))
elif opt.ood_dataset == 'CIFAR10':
    ood_test_dataset = datasets.CIFAR10('data', train=False, download=True,
                   transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    ]))

out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                        shuffle=False, num_workers=int(opt.workers)) 


criterion = nn.MSELoss()

def calc_p_values(test, cal):

    cal_reshaped = cal.reshape(1,-1) 
    test_reshaped = test.reshape(-1,1)

    compare = test_reshaped<=cal_reshaped

    p_values = np.sum(compare, axis=1)
    p_values = (p_values+1)/(len(cal)+1)

    return p_values

def bh(n,consts,p_vals):

    compare = np.array(p_vals<consts)*1
    compare = np.sum(compare, axis = 1)
    out = (compare > 0)*1

    return out

def get_all_stats(net, device, stats_list):

    flag = 0
    net.eval()

    if not os.path.exists('stats'):
        os.mkdir('stats')
    if not os.path.exists('stats/densenet'):
        os.mkdir('stats/densenet')
    if 'Mahalanobis' in stats_list:
        
        if not os.path.exists('stats/densenet/Mahalanobis'):
            os.mkdir('stats/densenet/Mahalanobis')
        
        if not os.path.exists('densenet_sample_class_mean.p'):
            sample_mean, precision = sample_estimator(net, device, train_dataloader, num_output)
            with open('densenet_sample_class_mean.p', 'wb') as f:
                pickle.dump(sample_mean, f)
            with open('densenet_precision.p', 'wb') as f:
                pickle.dump(precision, f)

        with open('densenet_sample_class_mean.p', 'rb') as f:
            sample_mean = pickle.load( f)
            f.close()
    
        with open('densenet_precision.p', 'rb') as f:
            precision = pickle.load( f)
            f.close()

        if not os.path.exists('stats/densenet/Mahalanobis/cal.npy'):
            temp_cal_stats = get_Mahalanobis_stats(net, device, cal_dataloader, sample_mean, precision, num_output)
            with open('stats/densenet/Mahalanobis/cal.npy', 'wb') as f:
                np.save(f, temp_cal_stats)
        else: 
            with open('stats/densenet/Mahalanobis/cal.npy', 'rb') as f:
                temp_cal_stats = np.load(f)

        if not os.path.exists('stats/densenet/Mahalanobis/in.npy'):
            temp_in_stats = get_Mahalanobis_stats(net, device, in_test_dataloader, sample_mean, precision, num_output)
            with open('stats/densenet/Mahalanobis/in.npy', 'wb') as f:
                np.save(f, temp_in_stats)
        else: 
            with open('stats/densenet/Mahalanobis/in.npy', 'rb') as f:
                temp_in_stats = np.load(f)

        if not os.path.exists('stats/densenet/Mahalanobis/ood_{}.npy'.format(opt.ood_dataset)):
            temp_ood_stats = get_Mahalanobis_stats(net, device, out_test_dataloader, sample_mean, precision, num_output)
            with open('stats/densenet/Mahalanobis/ood_{}.npy'.format(opt.ood_dataset), 'wb') as f:
                np.save(f, temp_ood_stats)
        else: 
            with open('stats/densenet/Mahalanobis/ood_{}.npy'.format(opt.ood_dataset), 'rb') as f:
                temp_ood_stats = np.load(f)

        if flag == 0:
            cal_set_stats = temp_cal_stats
            in_set_stats = temp_in_stats
            ood_set_stats = temp_ood_stats
            flag = 1
        else:
            cal_set_stats = np.concatenate((cal_set_stats, temp_cal_stats), axis = 1)
            in_set_stats = np.concatenate((in_set_stats, temp_in_stats), axis = 1)
            ood_set_stats = np.concatenate((ood_set_stats, temp_ood_stats), axis = 1)

    if 'gram' in stats_list or 'gram4' in stats_list or 'gram_all' in stats_list:
        if not os.path.exists('densenet_mins.p') or not os.path.exists('densenet_maxs.p'):
            mins, maxs = get_min_max_gram(net, device, train_dataloader, range(1,11), num_output)
            with open('densenet_mins.p', 'wb') as f:
                pickle.dump(mins, f)
                f.close()
            with open('densenet_maxs.p', 'wb') as f:
                pickle.dump(maxs, f)
                f.close()

        with open('densenet_mins.p', 'rb') as f:
            mins = pickle.load(f)
            f.close()
        
        with open('densenet_maxs.p', 'rb') as f:
            maxs = pickle.load(f)
            f.close()
        
        if not os.path.exists('stats/densenet/gram'):
            os.mkdir('stats/densenet/gram')

        if not os.path.exists('stats/densenet/gram/cal.npy'):
            cal_gram_stats = get_deviations_gram(net, device, cal_dataloader, mins, maxs, range(1,11), num_output)
            with open('stats/densenet/gram/cal.npy', 'wb') as f:
                np.save(f, cal_gram_stats)
        else: 
            with open('stats/densenet/gram/cal.npy', 'rb') as f:
                cal_gram_stats = np.load(f)

        if not os.path.exists('stats/densenet/gram/in.npy'):
            in_gram_stats = get_deviations_gram(net, device, in_test_dataloader, mins, maxs, range(1,11), num_output)
            with open('stats/densenet/gram/in.npy', 'wb') as f:
                np.save(f, in_gram_stats)
        else: 
            with open('stats/densenet/gram/in.npy', 'rb') as f:
                in_gram_stats = np.load(f)
        
        if not os.path.exists('stats/densenet/gram/ood_{}.npy'.format(opt.ood_dataset)):
            ood_gram_stats = get_deviations_gram(net, device, out_test_dataloader, mins, maxs, range(1,11), num_output)
            with open('stats/densenet/gram/ood_{}.npy'.format(opt.ood_dataset), 'wb') as f:
                np.save(f, ood_gram_stats)
        else: 
            with open('stats/densenet/gram/ood_{}.npy'.format(opt.ood_dataset), 'rb') as f:
                ood_gram_stats = np.load(f)
        
        mean_deviations_layers = np.mean(cal_gram_stats, axis = 0)
        cal_gram_stats = cal_gram_stats/mean_deviations_layers
        in_gram_stats = in_gram_stats/mean_deviations_layers
        ood_gram_stats = ood_gram_stats/mean_deviations_layers

        if 'gram4' in stats_list:
            cal_gram_stats = cal_gram_stats[:, 1:5]
            in_gram_stats = in_gram_stats[:, 1:5]
            ood_gram_stats = ood_gram_stats[:, 1:5]

        if 'gram_all' in stats_list:
            cal_gram_stats = np.sum(cal_gram_stats, axis = 1)
            in_gram_stats = np.sum(in_gram_stats, axis = 1)
            ood_gram_stats = np.sum(ood_gram_stats, axis = 1)
            cal_gram_stats = np.reshape(cal_gram_stats, (-1,1))
            in_gram_stats = np.reshape(in_gram_stats, (-1,1))
            ood_gram_stats = np.reshape(ood_gram_stats, (-1,1))

        if flag == 0:
            cal_set_stats = cal_gram_stats
            in_set_stats = in_gram_stats
            ood_set_stats = ood_gram_stats
            flag = 1
        else:
            cal_set_stats = np.concatenate((cal_set_stats, cal_gram_stats), axis = 1)
            in_set_stats = np.concatenate((in_set_stats, in_gram_stats), axis = 1)
            ood_set_stats = np.concatenate((ood_set_stats, ood_gram_stats), axis = 1)

    if 'energy' in stats_list:
        cal_energy_stats = get_energy_stats(net, device, cal_dataloader, opt.energy_temperature)
        in_energy_stats = get_energy_stats(net, device, in_test_dataloader, opt.energy_temperature)
        ood_energy_stats = get_energy_stats(net, device, out_test_dataloader, opt.energy_temperature)
        if flag == 0:
            cal_set_stats = cal_energy_stats
            in_set_stats = in_energy_stats
            ood_set_stats = ood_energy_stats
            flag = 1
        else:
            cal_set_stats = np.concatenate((cal_set_stats, cal_energy_stats), axis = 1)
            in_set_stats = np.concatenate((in_set_stats, in_energy_stats), axis = 1)
            ood_set_stats = np.concatenate((ood_set_stats, ood_energy_stats), axis = 1)



    return cal_set_stats, in_set_stats, ood_set_stats

def check_OOD_multiple_test(net, device, stats_list = ['Mahalanobis']):

    print('Running Multiple testing with {}'.format(stats_list))
    
    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, stats_list)
    # print(np.shape(cal_set_stats))
    # print(np.shape(in_set_stats))
    # print(np.shape(ood_set_stats))
    num_stats = np.shape(cal_set_stats)[1]
    in_p_values = np.zeros((np.shape(in_set_stats)[0],np.shape(in_set_stats)[1]))
    ood_p_values =  np.zeros((np.shape(ood_set_stats)[0],np.shape(ood_set_stats)[1]))
    for i in range(num_stats):
        in_p_values[:,i] = calc_p_values(in_set_stats[:,i], cal_set_stats[:,i])
        ood_p_values[:,i] = calc_p_values(ood_set_stats[:,i], cal_set_stats[:,i])

    in_p_values.sort(axis = 1)
    ood_p_values.sort(axis = 1)
    val_alphas = np.zeros((num_stats))
    for i in range(num_stats):
        temp_p_vals = np.sort(in_p_values[:,i])
        if stats_list == ['Mahalanobis']:
            val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.09)]*num_stats/(i+1)
        elif stats_list == ['gram']:
            val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.095)]*num_stats/(i+1)
        elif stats_list == ['Mahalanobis', 'energy']:
            val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.095)]*num_stats/(i+1)
        elif stats_list == ['gram', 'energy']:
            val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.095)]*num_stats/(i+1)
        elif stats_list == ['Mahalanobis', 'gram']:
            val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.09)]*num_stats/(i+1)
        elif stats_list == ['Mahalanobis', 'gram', 'energy']:
            val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.095)]*num_stats/(i+1)
        # if 'Mahalanobis' in stats_list:
        #     val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.085)]*num_stats/(i+1)
        # else: 
        #     val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*0.095)]*num_stats/(i+1)
    # print(val_alphas)
    alpha = min(val_alphas)
    consts = np.zeros((num_stats))
    for i in range(num_stats):
        consts[i] = alpha*(i+1)/num_stats
    # print(consts)
    ood_bh_output = bh(num_stats,consts,ood_p_values)
    tnr = np.mean(ood_bh_output)*100
    in_dist_bh_output = bh(num_stats,consts,in_p_values)
    fpr = np.mean(in_dist_bh_output)*100
    print(tnr)
    print(fpr)
    
    return tnr, fpr

def get_auroc_multiple_test(net, device, stats_list = ['Mahalanobis']):

    print('Getting Multiple testing AUROC with {}'.format(stats_list))
    tpr_list = []
    fpr_list = []
    threshold = np.linspace(99,1,100)/100.0
    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, stats_list)
    # print(np.shape(cal_set_stats))
    # print(np.shape(in_set_stats))
    # print(np.shape(ood_set_stats))
    num_stats = np.shape(cal_set_stats)[1]
    in_p_values = np.zeros((np.shape(in_set_stats)[0],np.shape(in_set_stats)[1]))
    ood_p_values =  np.zeros((np.shape(ood_set_stats)[0],np.shape(ood_set_stats)[1]))
    for i in range(num_stats):
        in_p_values[:,i] = calc_p_values(in_set_stats[:,i], cal_set_stats[:,i])
        ood_p_values[:,i] = calc_p_values(ood_set_stats[:,i], cal_set_stats[:,i])

    in_p_values.sort(axis = 1)
    ood_p_values.sort(axis = 1)
    for t in range(len(threshold)):   
        val_alphas = np.zeros((num_stats))
        for i in range(num_stats):
            temp_p_vals = np.sort(in_p_values[:,i])
            val_alphas[i] = temp_p_vals[int(len(temp_p_vals)*threshold[t])]*num_stats/(i+1)
        alpha = min(val_alphas)
        #alpha = 1
        consts = np.zeros((num_stats))
        for i in range(num_stats):
            consts[i] = alpha*(i+1)/num_stats
        ood_bh_output = bh(num_stats,consts,ood_p_values)
        fpr = 1.0 - np.mean(ood_bh_output)
        in_dist_bh_output = bh(num_stats,consts,in_p_values)
        tpr = 1.0 - np.mean(in_dist_bh_output)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    return fpr_list, tpr_list

def check_OOD_gram(net, device):
    
    print('Running gram statistics test')

    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, ['gram'])
    in_set_stats = np.sum(in_set_stats, axis = 1)
    ood_set_stats = np.sum(ood_set_stats, axis = 1)
    len_in_stats = np.shape(in_set_stats)[0]
    threshold = np.sort(in_set_stats)[int(0.9*len_in_stats)]
    tnr = np.mean(ood_set_stats > threshold)*100.0
    fpr = np.mean(in_set_stats > threshold)*100.0
    print(tnr)
    print(fpr)

    return tnr, fpr

def get_auroc_gram(net, device):

    print('Getting Gram AUROC')
    tpr_list = []
    fpr_list = []
    threshold = np.linspace(1,99,1000)/100.0
    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, ['gram'])
    in_set_stats = np.sum(in_set_stats, axis = 1)
    ood_set_stats = np.sum(ood_set_stats, axis = 1)
    len_in_stats = np.shape(in_set_stats)[0]
    
    for t in range(len(threshold)):   
        thresh = np.sort(in_set_stats)[int((threshold[t])*len_in_stats)]
        fpr = 1.0 - np.mean(ood_set_stats > thresh)
        tpr = 1.0 - np.mean(in_set_stats > thresh)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    return fpr_list, tpr_list


def check_OOD_energy(net, device):
    
    print('Running energy statistics test')
    
    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, ['energy'])
    in_set_stats = np.sum(in_set_stats, axis = 1)
    ood_set_stats = np.sum(ood_set_stats, axis = 1)
    len_in_stats = np.shape(in_set_stats)[0]
    threshold = np.sort(in_set_stats)[int(0.9*len_in_stats)]
    tnr = np.mean(ood_set_stats > threshold)*100.0
    fpr = np.mean(in_set_stats > threshold)*100.0
    print(tnr)
    print(fpr)
    
    return tnr, fpr

def get_auroc_energy(net, device):

    print('Getting Energy AUROC')
    tpr_list = []
    fpr_list = []
    threshold = np.linspace(1,99,1000)/100.0
    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, ['energy'])
    in_set_stats = np.sum(in_set_stats, axis = 1)
    ood_set_stats = np.sum(ood_set_stats, axis = 1)
    len_in_stats = np.shape(in_set_stats)[0]
    
    for t in range(len(threshold)):   
        thresh = np.sort(in_set_stats)[int((threshold[t])*len_in_stats)]
        fpr = 1.0 - np.mean(ood_set_stats > thresh)
        tpr = 1.0 - np.mean(in_set_stats > thresh)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    return fpr_list, tpr_list


def check_OOD_Mahalanobis(net, device, i):
    
    print('Running single Mahalanobis test')
    tnr_list = []
    fpr_list = []

    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, ['Mahalanobis'])
    in_set_stats = in_set_stats[:,i]
    ood_set_stats = ood_set_stats[:,i]
    len_in_stats = np.shape(in_set_stats)[0]
    threshold = np.sort(in_set_stats)[int(0.9*len_in_stats)]
    tnr = np.mean(ood_set_stats > threshold)*100.0
    fpr = np.mean(in_set_stats > threshold)*100.0
    print(tnr)
    print(fpr)

    return tnr, fpr

def get_auroc_Mahalanobis(net, device):

    print('Getting Mahalanobis AUROC')
    tpr_list = []
    fpr_list = []
    threshold = np.linspace(1,99,1000)/100.0
    cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, ['Mahalanobis'])
    in_set_stats = in_set_stats[:,num_output-1]
    ood_set_stats = ood_set_stats[:,num_output-1]
    len_in_stats = np.shape(in_set_stats)[0]
    
    for t in range(len(threshold)):   
        thresh = np.sort(in_set_stats)[int((threshold[t])*len_in_stats)]
        fpr = 1.0 - np.mean(ood_set_stats > thresh)
        tpr = 1.0 - np.mean(in_set_stats > thresh)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    return fpr_list, tpr_list

def get_false_alarm_multiple(net, device, stats_list = ['Mahalanobis']):

    # print('Running Multiple testing with {}'.format(stats_list))
    tnr_list = []
    fpr_list = []
    
    alpha_list = np.arange(0.01,1,0.01)

    # alpha_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    for a in alpha_list:
        # print(a)
        sum_K = 0
        for i in range(len(stats_list)):
            sum_K += 1/(1+i)
        alpha_prime = a / (2 * sum_K)
        # print(alpha_prime)
        
        cal_set_stats, in_set_stats, ood_set_stats = get_all_stats(net, device, stats_list)
        num_stats = np.shape(cal_set_stats)[1]
        in_p_values = np.zeros((np.shape(in_set_stats)[0],np.shape(in_set_stats)[1]))
        ood_p_values =  np.zeros((np.shape(ood_set_stats)[0],np.shape(ood_set_stats)[1]))
        for i in range(num_stats):
            in_p_values[:,i] = calc_p_values(in_set_stats[:,i], cal_set_stats[:,i])
            ood_p_values[:,i] = calc_p_values(ood_set_stats[:,i], cal_set_stats[:,i])

        in_p_values.sort(axis = 1)
        ood_p_values.sort(axis = 1)
        val_alphas = np.zeros((num_stats))
        alpha = alpha_prime
        # print(alpha)
        consts = np.zeros((num_stats))
        for i in range(num_stats):
            consts[i] = alpha*(i+1)/num_stats
                #consts[i] = max(alpha*(i+1)/n,val_alphas[i]*(i+1)/n)
        
        ood_bh_output = bh(num_stats,consts,ood_p_values)
        tnr = np.mean(ood_bh_output)*100
        in_dist_bh_output = bh(num_stats,consts,in_p_values)
        fpr = np.mean(in_dist_bh_output)*100
        # print(tnr)
        # print(fpr)
        tnr_list.append(tnr)
        fpr_list.append(fpr)

    with open('SVHN_DenseNet_FPR.npy', 'wb') as f:
        np.save(f, fpr_list)
    return tnr_list, fpr_list



if __name__ == "__main__":

    #Getting TPR with theoretical thresholds
    # tnr_list, fpr_list = get_false_alarm_multiple(net, device, stats_list = ['Mahalanobis', 'gram', 'energy'])
    
    #Getting TNR results
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/DenseNet_SVHN.csv'):
        with open("results/DenseNet_SVHN.csv", "w") as f:
            wf = writer(f)
            wf.writerow(['method', 'stats list', opt.ood_dataset, 'fpr','tnr', 'other params'])
            f.close()
    # Mahalanobis baseline
    tnr_list, fpr_list = check_OOD_Mahalanobis(net, device, num_output - 1)
    with open("results/DenseNet_SVHN.csv", "a") as f:
        wf = writer(f)
        wf.writerow(['Mahalanobis', ['Mahalanobis'], opt.ood_dataset, np.mean(np.array(fpr_list)),np.mean(np.array(tnr_list)), None])
        f.close()
    # Energy baseline
    tnr_list, fpr_list = check_OOD_energy(net, device)
    with open("results/DenseNet_SVHN.csv", "a") as f:
         wf = writer(f)
         wf.writerow(['energy', ['energy'], opt.ood_dataset, np.mean(np.array(fpr_list)),np.mean(np.array(tnr_list)), None])
         f.close()
    # Gram baseline
    tnr_list, fpr_list = check_OOD_gram(net, device)
    with open("results/DenseNet_SVHN.csv", "a") as f:
         wf = writer(f)
         wf.writerow(['gram', ['gram'], opt.ood_dataset, np.mean(np.array(fpr_list)),np.mean(np.array(tnr_list)), None])
         f.close()
    # Note that the threshold in line 327 and 329 needs to be changed for each combination to control TPR appropriately 
    #Proposed method with all possible combinations
    list_of_stats_list = [['Mahalanobis'], ['gram'], ['Mahalanobis', 'energy'], ['gram', 'energy'], ['Mahalanobis', 'gram'], ['Mahalanobis', 'gram','energy']]
    for stats_list in list_of_stats_list:
        tnr_list, fpr_list = check_OOD_multiple_test(net, device, stats_list)
        with open("results/DenseNet_SVHN.csv", "a") as f:
            wf = writer(f)
            wf.writerow(['multiple', stats_list, opt.ood_dataset, np.mean(np.array(fpr_list)),np.mean(np.array(tnr_list)), 'none'])
            f.close()
    
    #Getting AUROC results
    if not os.path.exists('results/auroc_Densenet_SVHN.csv'):
        with open("results/auroc_Densenet_SVHN.csv", "w") as f:
            wf = writer(f)
            wf.writerow(['method', 'stats list', 'ood dataset', 'auroc'])
            f.close()
    # Mahalanobis baseline
    fpr_list, tpr_list = get_auroc_Mahalanobis(net, device)
    auroc = np.trapz(1.0 - np.array([0.] + fpr_list), [0.] + tpr_list)
    with open("results/auroc_Densenet_SVHN.csv", "a") as f:
        wf = writer(f)
        wf.writerow(['Mahalanobis', 'Mahalanobis', opt.ood_dataset, auroc])
        f.close()
    #Gram baseline
    fpr_list, tpr_list = get_auroc_gram(net, device)
    auroc = np.trapz(1.0 - np.array([0.] +fpr_list), [0.] +tpr_list)
    with open("results/auroc_Densenet_SVHN.csv", "a") as f:
        wf = writer(f)
        wf.writerow(['gram', 'gram', opt.ood_dataset, auroc])
        f.close()
    # Energy baseline
    fpr_list, tpr_list = get_auroc_energy(net, device)
    auroc = np.trapz(1.0 - np.array([0.] +fpr_list),[0.] + tpr_list)
    with open("results/auroc_Densenet_SVHN.csv", "a") as f:
        wf = writer(f)
        wf.writerow(['Energy', 'Energy', opt.ood_dataset, auroc])
        f.close()
    # Proposed method with all possible combinations
    list_of_stats_list = [['Mahalanobis'], ['gram'], ['Mahalanobis', 'energy'], ['gram', 'energy'], ['Mahalanobis', 'gram'], ['Mahalanobis', 'gram','energy']]
    for stats_list in list_of_stats_list:
        fpr_list, tpr_list = get_auroc_multiple_test(net, device, stats_list)
        # plt.plot(fpr_list, tpr_list)
        # plt.savefig('temp.png')
        auroc = np.trapz(1.0 - np.array([0.] + fpr_list), [0.] + tpr_list)
        print(auroc)
        with open("results/auroc_Densenet_SVHN.csv", "a") as f:
            wf = writer(f)
            wf.writerow(['Multiple', stats_list, opt.ood_dataset, auroc])
            f.close()
