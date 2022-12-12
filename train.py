import os
import sys
import torch
import itertools
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models import *
from utilities import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Train_Models (model_alpha,
                  model_beta,
                  model_encoder,
                  model_transformer,
                  model_classifier,
                  train_list, valid_list,
                  X_train_list, X_valid_list,
                  criterion, optimizer1, optimizer2, scheduler,
                  epochs = 100, cut_epochs = 50, h1 = 3200, w1 = 3200,
                  h2 = 100, w2 = 100, s1 = 0.03125, s2 = 0.125, N = 100) :
    list_loss_train = []
    list_loss_valid = []
    list_auc_train = []
    list_auc_valid = []
    list_acc_train = []
    list_acc_valid = []
    ds_train_all = np.zeros((epochs, len(train_list), 2))
    ds_valid_all = np.zeros((epochs, len(valid_list), 2))
    for epoch in range(0, epochs) :
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 60)
        for phase in ["train", "validation"] :
            num_correct = 0
            if phase == "train" :
                curr_list, curr_X_list = train_list, X_train_list
                model_alpha      = model_alpha.train()
                model_beta       = model_beta.train()
                model_encoder    = model_encoder.train()
                model_classifier = model_classifier.train()
                if epoch >= cut_epochs :
                    model_transformer = model_transformer.train(True)
                else :
                    model_transformer = model_transformer.train(False)
            else :
                curr_list, curr_X_list = valid_list, X_valid_list
                model_alpha       = model_alpha.eval()
                model_beta        = model_beta.eval()
                model_encoder     = model_encoder.eval()
                model_transformer = model_transformer.eval()
                model_classifier  = model_classifier.eval()
            random_idx = random_shuffle_data(curr_list)
            loss_epoch = 0
            ds_idx_count = 0
            ds_epoch = np.zeros((len(random_idx), 2))
            for each_idx in random_idx :
                img_path = curr_list[each_idx]
                X = np.load(img_path)
                # Incremental Testing
                # X = curr_X_list[each_idx]
                pad_flag_X = 0
                Y = decide_true_label(img_path, label_num = 1)
                Y = torch.Tensor(Y).to(device)
                Xs1 = cv.resize(X, (0, 0), fx = s1, fy = s1)
                Xs1 = np.reshape(Xs1, (3, Xs1.shape[0], Xs1.shape[1]))
                Xs1 = np.expand_dims(Xs1, axis = 0)
                Xs1 = torch.Tensor(Xs1).to(device)
                out_a = model_alpha(Xs1)
                s1_col = out_a.shape[1]
                out_a = torch.flatten(out_a)
                s1_idx = np.linspace(start = 0, stop = out_a.shape[0] - 1,
                                     num = out_a.shape[0]).astype(int)
                out_alpha = F.softmax(out_a, dim = 0)
                XB = np.zeros((N, 3, h2, w2))
                XB_count = 0
                loss_beta = torch.Tensor([0]).to(device)
                s1_samples = np.random.choice(a = s1_idx, size = N,
                                replace = True,
                                p = out_alpha.cpu().detach().numpy())
                s1_samples = np.sort(s1_samples)
                s1_tuples = [(x,len(list(y))) for x, y in itertools.groupby(s1_samples)]
                print(img_path, len(s1_tuples))
                for x in s1_tuples :
                    i, j = int(x[0]//s1_col), int(x[0]%s1_col)
                    if pad_flag_X == 0 :
                        pad_flag_X = 1
                        X = np.pad(X, pad_width = ((0, h1), (0, w1), (0, 0)),
                                   mode = "constant", constant_values = 0)
                    Xa = np.copy(X[int(i/s1):int(i/s1+h1), int(j/s1):int(j/s1+w1), :])
                    pad_flag_Xa = 0
                    Xas2 = cv.resize(Xa, (0, 0), fx = s2, fy = s2)
                    Xas2 = np.reshape(Xas2, (3, Xas2.shape[0], Xas2.shape[1]))
                    Xas2 = np.expand_dims(Xas2, axis = 0)
                    Xas2 = torch.Tensor(Xas2).to(device)
                    out_axb = model_beta(Xas2)
                    s2_col = out_axb.shape[1]
                    out_axb = torch.flatten(out_axb)
                    s2_idx = np.linspace(start = 0, stop = out_axb.shape[0] - 1,
                                         num = out_axb.shape[0]).astype(int)
                    out_a_beta = F.softmax(out_axb, dim = 0)
                    loss_beta = loss_beta + x[1] * er_beta(out_a_beta)
                    s2_samples = np.random.choice(a = s2_idx, size = x[1],
                                                  replace = False,
                                                  p = out_a_beta.cpu().detach().numpy())
                    s2_samples = np.sort(s2_samples)
                    for s2_x in s2_samples :
                        s2_i, s2_j = int(s2_x//s2_col), int(s2_x%s2_col)
                        # Visualize the sampled tile and sub-tile indices
                        # print((i, j), (s2_i, s2_j))
                        if pad_flag_Xa == 0 :
                            pad_flag_Xa = 1
                            Xa = np.pad(Xa, pad_width = ((0, h2), (0, w2), (0, 0)),
                                        mode = "constant", constant_values = 0)
                        Xb = np.copy(Xa[int(s2_i/s2):int(s2_i/s2+h2),
                                        int(s2_j/s2):int(s2_j/s2+w2), :])
                        Xb = np.reshape(Xb, (3, Xb.shape[0], Xb.shape[1]))
                        XB[XB_count, :, :, :] = np.copy(Xb)
                        XB_count = XB_count + 1
                XB = torch.Tensor(XB).to(device)
                out_z = model_encoder(XB)

                #################################################################
                if epoch < cut_epochs :
                    # Do Not Apply Transformer Here!
                    out_z = (1 / N) * torch.sum(out_z, dim = 0)
                    y_pred_ds = model_classifier(out_z, epoch)
                else :
                    # Apply Transformer Here!
                    Q_K, out_t = model_transformer(out_z)
                    ##################################################
                    # Introduce non-linearity to avoid collapsing
                    # out_t = F.relu(out_t)
                    # This is not working!
                    ##################################################
                    out_t = (1 / N) * torch.sum(out_t, dim = 0)
                    y_pred_ds = model_classifier(out_t, epoch)
                    if each_idx < 2 :
                        print(Q_K[0:2])
                #################################################################

                if torch.equal(torch.round(y_pred_ds), Y) :
                    num_correct = num_correct + 1
                print("y = %.0f" % Y[0], "y_pred = %.3f" % y_pred_ds[0])

                loss = criterion(y_pred_ds, Y)
                loss = loss + er_alpha(out_alpha) + loss_beta

                loss_epoch += loss.item()
                ds_epoch[ds_idx_count, 0] = Y.cpu().detach().numpy()[0]
                ds_epoch[ds_idx_count, 1] = y_pred_ds.cpu().detach().numpy()[0]
                ds_idx_count = ds_idx_count + 1
                if phase == "train" :
                    if epoch >= cut_epochs :
                        optimizer1.zero_grad()
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer1.step()
                        optimizer2.step()
                    else :
                        optimizer1.zero_grad()
                        loss.backward()
                        optimizer1.step()
            ##########################################################
            if phase == "train" :
                ds_train_all[epoch, :, :] = np.copy(ds_epoch)
            if phase == "validation" :
                ds_valid_all[epoch, :, :] = np.copy(ds_epoch)
            ##########################################################
            s1_pd, s1_pfa = compute_ROC(ds_epoch, flex_type = 1)
            auc1 = metrics.auc(s1_pfa, s1_pd)
            epoch_acc = float(num_correct) / len(curr_list)
            loss_epoch = loss_epoch / len(curr_list)
            if phase == "train" :
                print("Current Epoch Training Loss     = %.3f" % loss_epoch)
                print("Current Epoch Training AUC      = %.3f" % auc1)
                print("Current Epoch Training Accuracy = %.3f" % epoch_acc)
                list_loss_train.append(loss_epoch)
                list_auc_train.append(auc1)
                list_acc_train.append(epoch_acc)
            if phase == "validation" :
                print("Current Epoch Validation Loss     = %.3f" % loss_epoch)
                print("Current Epoch Validation AUC      = %.3f" % auc1)
                print("Current Epoch Validation Accuracy = %.3f" % epoch_acc)
                list_loss_valid.append(loss_epoch)
                list_auc_valid.append(auc1)
                list_acc_valid.append(epoch_acc)
        scheduler.step()
    print("Training and Validation Complete!")
    tr_loss, tr_auc = np.asarray(list_loss_train), np.asarray(list_auc_train)
    va_loss, va_auc = np.asarray(list_loss_valid), np.asarray(list_auc_valid)
    tr_acc, va_acc  = np.asarray(list_acc_train),  np.asarray(list_acc_valid)
    return tr_loss, tr_auc, tr_acc, ds_train_all, va_loss, va_auc, va_acc, ds_valid_all
