import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Some customized helper functions for plotting ROC curves

def divide_labels (arr_data) :
    alldat = np.copy(arr_data)
    dat_zero = alldat[alldat[:,0]==0, :]
    dat_one = alldat[alldat[:,0]==1, :]
    return dat_zero, dat_one

def compute_PFA (H0, thres) :
    H0_ds = np.sort(H0[:, 1])
    false_alarm = float(len(H0_ds[H0_ds >= thres])) / (H0.shape[0])
    return false_alarm

def compute_PD (H1, thres) :
    H1_ds = np.sort(H1[:, 1])
    detection = float(len(H1_ds[H1_ds >= thres])) / (H1.shape[0])
    return detection

def flex_thresholds (arr_data, flex_type) :
    alldata = np.copy(arr_data)

    if flex_type == 1 :
        # every decision statistics as a threshold
        thres = np.zeros(2 + alldata.shape[0])
        thres[1:-1] = np.sort(alldata[:, 1])
        thres[0] = float("-inf")
        thres[-1] = float("inf")

    if flex_type == 2 :
        # thresholds selected so they linearly
        # sample the range of decision statistics
        # 99 linearly spaced samples from min to max
        # plus -inf and +inf
        thres = np.zeros(101)
        lam_min = np.min(alldata[:, 1])
        lam_max = np.max(alldata[:, 1])
        thres[1:-1] = np.linspace(lam_min, lam_max, num = 99)
        thres[0] = float("-inf")
        thres[-1] = float("inf")

    return thres

def compute_ROC (arr_data, flex_type) :
    arr_thres = flex_thresholds(arr_data, flex_type)
    arr_PD = np.zeros(len(arr_thres))
    arr_PFA = np.zeros(len(arr_thres))
    H0, H1 = divide_labels(arr_data)
    for i in range(len(arr_thres)) :
        arr_PD[i] = compute_PD(H1, arr_thres[i])
        arr_PFA[i] = compute_PFA(H0, arr_thres[i])
    return np.flipud(arr_PD), np.flipud(arr_PFA)

# Some customized helper functions for data preprocessing

def build_all_training_file_list (N_normal, N_tumor, dir1 = "training",
                                  file_suffix = ".npy") :
    results_normal = []
    results_tumor = []
    # Append Normal List
    init_path = dir1 + "/" + "normal" + "/" + "normal" + "_"
    for i in range(1, N_normal + 1) :
        if i == 86 :
            continue
        else :
            if i < 10 :
                curr_path = init_path + "00" + str(i) + file_suffix
            elif i < 100 :
                curr_path = init_path + "0" + str(i) + file_suffix
            else :
                curr_path = init_path + str(i) + file_suffix
            results_normal.append(curr_path)
    # Append Tumor List
    init_path = dir1 + "/" + "tumor" + "/" + "tumor" + "_"
    for i in range(1, N_tumor + 1) :
        if i < 10 :
            curr_path = init_path + "00" + str(i) + file_suffix
        elif i < 100 :
            curr_path = init_path + "0" + str(i) + file_suffix
        else :
            curr_path = init_path + str(i) + file_suffix
        results_tumor.append(curr_path)
    results = results_normal + results_tumor
    return results

def build_all_testing_file_list (N_test, dir1 = "testing",
                                 file_suffix = ".npy") :
    results = []
    # Append Testing List
    init_path = dir1 + "/" + "test" + "_"
    for i in range(1, N_test + 1) :
        if i == 49 :
            continue
        else :
            if i < 10 :
                curr_path = init_path + "00" + str(i) + file_suffix
            elif i < 100 :
                curr_path = init_path + "0" + str(i) + file_suffix
            else :
                curr_path = init_path + str(i) + file_suffix
            results.append(curr_path)
    return results

def preprocess_data (img_mr, img_path, level = 3) :
    image_shape = img_mr.getLevelDimensions(level)
    image_array = img_mr.getUCharPatch(0, 0, image_shape[0],
                                             image_shape[1], level).astype("uint8")
    return image_array

# Some customized helper functions for network optimization

def random_shuffle_data (curr_list) :
    slide_num = len(curr_list)
    slide_idx = np.linspace(start = 0, stop = slide_num - 1,
                            num = slide_num).astype(int)
    random_idx = np.random.choice(a = slide_idx, size = slide_num,
                                  replace = False, p = None)
    return random_idx

def decide_true_label(img_path, label_num) :
    if "normal" in img_path :
        Y = np.zeros(label_num)
    else :
        Y = np.ones(label_num)
    return Y

def er_alpha (x) :
    y = 1e-5 * torch.sum(torch.mul(x, torch.clamp(torch.log(x), min = -100)))
    return y

def er_beta (x) :
    y = 1e-5 * torch.sum(torch.mul(x, torch.clamp(torch.log(x), min = -100)))
    return y
