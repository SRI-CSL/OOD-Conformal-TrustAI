from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
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


import pdb

from scipy.integrate import quad_vec
from scipy import integrate
import sklearn.covariance

num_classes =100

def get_intermediate_features(net, device, data_loader, num_output):
    num_classes = 100
    correct, total = 0, 0
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    #num_output = len(feature_name_list)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    #Get all features in feature_name_list
    for _, data in enumerate(data_loader, 0):
        img1 = data[0].to(device)
        labels = data[1]
        total += data[0].size(0)
        y, out_features = net.feature_list(img1)
        
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            out_features[i] = out_features[i].cpu().detach().numpy()
        
        for i in range(data[0].size(0)):
            label = labels[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = np.reshape(out[i],(1, -1))
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = np.concatenate((list_features[out_count][label], np.reshape(out[i],(1, -1))),axis = 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
    
    #Get the sizes of features 
    feature_list = []
    temp_sum = 0
    for j in range(num_output):
        feature_list.append(np.shape(list_features[j][0])[1])
    #print(feature_list)

    return list_features, feature_list

def G_p(ob, p):
    temp = torch.from_numpy(ob)
    temp = temp**p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0 = 2, dim1 = 1)))).sum(dim = 2)
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0], -1)
    return temp

def get_min_max_gram(net, device, data_loader, power, num_output):
    mins = []
    maxs = []
    #num_output = len(feature_name_list)
    num_classes = 100
    net.eval()

    list_features, feature_list = get_intermediate_features(net, device, data_loader, num_output) 

    for c in range(num_classes):
        mins_class = []
        maxs_class = []
        for l in range(num_output):
            if l == len(mins_class):
                mins_class.append([None]*len(power))
                maxs_class.append([None]*len(power))
            
            for p, P in enumerate(power):
                g_p = G_p(list_features[l][c], P)

                current_min = g_p.min(dim = 0, keepdim = True)[0]
                current_max = g_p.max(dim = 0, keepdim = True)[0]
                if mins_class[l][p] is None:
                    mins_class[l][p] = current_min
                    maxs_class[l][p] = current_max
                else:
                    mins_class[l][p] = torch.min(current_min, mins_class[l][p])
                    maxs_class[l][p] = torch.max(current_max, maxs_class[l][p])

        mins.append(mins_class)
        maxs.append(maxs_class)

    # with open('mins.p', 'wb') as f:
    #     pickle.dump(mins, f)
    #     f.close()
    # with open('maxs.p', 'wb') as f:
    #     pickle.dump(maxs, f)
    #     f.close()

    return mins, maxs

def get_deviations_gram(net, device, data_loader, mins, maxs, power, num_output):

    preds = []
    num_classes  =100
    confs = []
    for _, data in enumerate(data_loader, 0):
        img1 = data[0].to(device)
        labels = data[1]
        logits = net(img1)
        scores = F.softmax(logits,dim=1).cpu().detach().numpy()
        predictions = np.argmax(scores,axis=1)
        confs.extend(np.max(scores,axis=1))
        preds.extend(predictions)  
        torch.cuda.empty_cache()

    all_deviations = []

    data_ctr = 0
    for index, data in enumerate(data_loader, 0):
        # print(index)
        img1 = data[0].to(device)
        labels = data[1]
        out_features = []
        for i in range(num_output):
            features = net.intermediate_forward(img1,i)
            features = features.view(features.size(0), features.size(1), -1)
            features = torch.mean(features.data, 2)
            out_features.append(features.cpu().detach().numpy())
        torch.cuda.empty_cache()
        for i in range(np.shape(out_features[0])[0]):
            pred_i = int(preds[data_ctr])
            data_ctr += 1
            dev_i = []
            for l in range(num_output):
                dev = 0
                for p, P in enumerate(power):
                    g_p = G_p(np.reshape(out_features[l][i],(1, out_features[l][i].shape[0])), P)
                    dev += (F.relu(mins[pred_i][l][p]-g_p)/torch.abs(mins[pred_i][l][p]+10**-6)).sum(dim=1,keepdim=True)
                    dev += (F.relu(g_p-maxs[pred_i][l][p])/torch.abs(maxs[pred_i][l][p]+10**-6)).sum(dim=1,keepdim=True)   
                dev = dev.cpu().detach().numpy()
                dev = dev[0][0]
                dev_i.append(dev)
            all_deviations.append(dev_i)
        
    all_deviations = np.array(all_deviations)
    #print(np.shape(all_deviations))

    return all_deviations

def sample_estimator(net, device, data_loader, num_output):
    
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    from sklearn.covariance import MinCovDet
    num_classes = 100
    #num_output = len(feature_name_list)
    list_features, feature_list = get_intermediate_features(net, device, data_loader, num_output)   
    
    #Get class-wise means for all features
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = np.zeros((num_classes, int(num_feature)))
        for j in range(num_classes):
            temp_list[j] = np.mean(list_features[out_count][j], axis = 0)
        sample_class_mean.append(temp_list)
        out_count += 1
    
    #Get common precision across all classes using sklearn.covariance.EmpiricalCovariance
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = np.concatenate((X, list_features[k][i] - sample_class_mean[k][i]), axis =  0)
                
        # find inverse            
        group_lasso.fit(X)
        # cov = MinCovDet(random_state=0).fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        # temp_precision = cov.precision_
        temp_precision = temp_precision
        precision.append(temp_precision)

    #TODO: Try computing sigma inverse using mincovdet

    # with open('sample_class_mean.p', 'wb') as f:
    #     pickle.dump(sample_class_mean, f)
    #     f.close()
    
    # with open('precision.p', 'wb') as f:
    #     pickle.dump(precision, f)
    #     f.close()

    return sample_class_mean, precision
    
def get_Mahalanobis_distance(net, device, data_loader, sample_mean, precision, layer_index):

    net.eval()
    layer_index = int(layer_index)
    Mahalanobis = []
    num_classes= 100
    #feat_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'classifier']
    #feat_layers = [ 'conv2', 'conv3', 'conv4', 'conv5', 'classifier']

    for _, data in enumerate(data_loader, 0):
        img1 = data[0].to(device)
        labels = data[1]
        out_features = net.intermediate_forward(img1, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        out_features = out_features.cpu().detach().numpy()
        #No input-preprocessing for now
        noise_out_features = out_features
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = 0.5*np.matmul(np.matmul(zero_f, precision[layer_index]), zero_f.T).diagonal()
            if i == 0:
                noise_gaussian_score = np.reshape(term_gau, (-1,1))
            else:
                noise_gaussian_score = np.concatenate((noise_gaussian_score, np.reshape(term_gau, (-1,1))), axis = 1)      

        noise_gaussian_score = np.min(noise_gaussian_score, axis=1)
        Mahalanobis.extend(noise_gaussian_score)

    return Mahalanobis

def get_Mahalanobis_stats(net, device, data_loader, sample_mean, precision, num_output):

    #feat_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'classifier']
    #feat_layers = ['conv2', 'conv3', 'conv4', 'conv5', 'classifier']
    #num_output = len(feat_layers)
    for i in range(num_output):
        M_in = get_Mahalanobis_distance(net, device, data_loader, sample_mean, precision, i)
        M_in = np.asarray(M_in, dtype=np.float32)
        if i == 0:
            Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
        else:
            Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

    Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
    
    return Mahalanobis_in


def get_energy_stats(net, device, data_loader, T):
    energy_stats = []
    for _, data in enumerate(data_loader, 0):
        img1 = data[0].to(device)
        labels = data[1]
        logits = net(img1)
        temp = -T*(torch.logsumexp(logits / T, dim=1).cpu().detach().numpy())
        temp = temp.tolist()
        energy_stats.extend(temp)
    
    energy_stats = np.array(energy_stats, ndmin = 2).T
    #print(np.shape(energy_stats))

    return energy_stats