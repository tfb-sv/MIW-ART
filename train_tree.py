########################################################################################
########################################################################################
########################################################################################

import argparse
import logging
import os
import time
import shutil
from collections import defaultdict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import softmax
from model.tree_model import ARTM_model
from tree_data import data_loaderX
from eval_tree import eval_iter
from tqdm import tqdm
import sys
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import wandb
import itertools
torch.manual_seed(0)
CUDA_VISIBLE_DEVICES=1

########################################################################################
########################################################################################

def train_iter(args, batch, model, params, criterion, optimizer):
    #######################################################################################
    model.train(True)
    ##########################
    model_arg, labels = batch
    ##########################
    logits, _ = model(**model_arg)
    ##########################
    if args.task == "clf":
        labels_pred = logits.max(1)[1]
        num_correct = torch.eq(labels, labels_pred).long().sum().item()       
    # elif args.task == "reg":
    #     logits = logits.view(-1)
    ##########################
    loss = criterion(input=logits, target=labels)
    ##########################
    optimizer.zero_grad()
    loss.backward()
    ##########################
    if args.is_clip:   # NRL
        clip_grad_norm_(parameters=params, max_norm=args.clip) 
    ##########################
    optimizer.step()
    ##########################
    if args.task == "clf":
        labelsx = labels.cpu().detach().numpy().tolist()
        labels_predx = labels_pred.cpu().detach().numpy().tolist()
        return loss, num_correct, labelsx, labels_predx
    # elif args.task == "reg":
    #     return loss
    #######################################################################################

########################################################################################
########################################################################################

def train(args, cnt, cv_keyz, data):
    ########################################################################################
    # prmz = {"batch_size": args.batch_size,
    #         "tree_hidden_dim": args.tree_hidden_dim,
    #         "dropout": args.dropout,
    #         "tokenization": args.tokenization}
    prmz = {cv_keyz[i]: getattr(args, cv_keyz[i]) for i in range(len(cv_keyz))}
    ##########################
    project_name = "ART-Mol" + "_" + args.data_name.upper()
    if args.tokenization == "cha":
        run_name = "cha_" + str(cnt)
    elif args.tokenization == "bpe":
        run_name = "bpe_" + str(cnt)
    token_note = args.tokenization
    wandb.init(mode=args.wandb_mode, project=project_name, config=prmz, name=run_name, notes=token_note)
    ##########################
    num_train_batches = data.num_train_batches
    num_valid_batches = data.num_valid_batches
    ########################################################################################
    modelX = ARTM_model
    model = modelX(args)
    ########################################################################################
    # if args.is_pretrain != "":
        # model.load_state_dict(torch.load(args.pre_train), strict=False)   # strict, 1e1 eşleşmezse hata vermesini engelliyor, isim eşleşmesine bakıyor
        # for name, param in model.named_parameters():
            # if name.startswith("classifier."):
                # param.requires_grad = False   # güncellememesi için bu layerları !!
    ##########################
    if args.fix_word_embedding:   # NRL
        model.word_embedding.weight.requires_grad = False 
    ##########################
    model = model.to(args.device)
    ##########################
    print("\n")
    logging.info(model)
    print("\n")
    ##########################
    params = [p for p in model.parameters() if p.requires_grad]  
    ########################################################################################
    # if args.optimizer == 'adam':
    optimizer_class = optim.Adam
    # elif args.optimizer == 'adagrad':
    #     optimizer_class = optim.Adagrad
    # elif args.optimizer == 'adadelta':
    #     optimizer_class = optim.Adadelta
    # else:
    #     raise Exception('unknown optimizer')  
    ##########################
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)
    if args.is_scheduler:   # NRL
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=args.patience, verbose=True) 
    ##########################
    if args.task == "clf":
        criterion = nn.CrossEntropyLoss()
        best_metric = 0
    elif args.task == "reg":
        criterion = nn.MSELoss()
        best_metric = 10
    ##########################
    trpack = [model, params, criterion, optimizer]
    evpack = []
    ##########################
    loss_outputs = {"ce loss train": [],
                    "ce loss valid": [],
                    "roc-auc": [],
                    "prc-auc": [],
                    "accuracy": []}
    ##########################
    # wandb.watch(model, log_freq=5)
    ########################################################################################
    ########################################################################################
    for epoch_num in range(args.max_epoch):
        train_loss_list, val_loss_list = [], []
        with tqdm(total=(num_train_batches), unit="batch") as pbar_train:                      
            for batch_iter, (train_batch) in enumerate(data.generator("train")):
                if args.task == "clf":
                    train_loss, curr_correct, labels, preds = train_iter(args, train_batch, *trpack)
                # elif args.task == "reg":
                #     train_loss = train_iter(args, train_batch, *trpack) 
                ##########################
                train_loss_list.append(train_loss.item())   
                pbar_train.set_description(f'>>  EPOCH {(epoch_num + 1)}T  |  CE Loss = {train_loss.item():.4f}  |')
                pbar_train.update()
                ########################################################################################
                ########################################################################################
                if (batch_iter + 1) % num_train_batches == 0:                  
                    with tqdm(total=(num_valid_batches), unit="batch") as pbar_val:
                        total_correct = 0
                        predictions, ground_truth = [], []
                        ##########################
                        for valid_batch in data.generator("valid"):                           
                            if args.task == "clf":
                                _, val_loss, curr_correct, labels, preds = eval_iter(valid_batch, model, args.task, args.mode)   # *evpack)
                                total_correct += curr_correct
                                predictions.extend(preds)
                                ground_truth.extend(labels)
                            ##########################
                            # elif args.task == "reg": 
                            #     _, val_loss = eval_iter(valid_batch, model, args.task)   # *evpack)
                            val_loss_list.append(val_loss.item())
                        ######################################################################################## CALCULATE COMMON METRICS
                        train_loss_mean = np.round(torch.mean(torch.Tensor(train_loss_list)).item(), 4)
                        valid_loss_mean = np.round(torch.mean(torch.Tensor(val_loss_list)).item(), 4)
                        ######################################################################################## PROCESSES FOR CLF             
                        if args.task == "clf":
                            roc_score = np.round(roc_auc_score(ground_truth, predictions), 4)
                            precision, recall, _ = precision_recall_curve(ground_truth, predictions)
                            prc_score = np.round(auc(recall, precision), 4)
                            valid_accuracy = np.round((total_correct / data.valid_size), 4)
                            ##########################
                            main_metric = roc_score  
                            ##########################
                            if main_metric > best_metric:
                                best_metric = main_metric
                                model_filename = (f'm-{args.data_name}-{main_metric:.4f}-{cnt}.pkl')
                                model_path = args.save_dir + "/" + args.data_name + "/" + model_filename
                                torch.save(model.state_dict(), model_path)
                                pbar_val.set_description(f'>>  EPOCH {(epoch_num + 1)}V  |  ROC-AUC = {roc_score:.4f}  |  CE Loss = {valid_loss_mean:.4f}  |  MODEL SAVED.  |')
                                pbar_val.update()
                            else:
                                pbar_val.set_description(f'>>  EPOCH {(epoch_num + 1)}V  |  ROC-AUC = {roc_score:.4f}  |  CE Loss = {valid_loss_mean:.4f}  |')
                                pbar_val.update()
                        ######################################################################################## PROCESSES FOR REG  
                        # elif args.task == "reg": 
                        #     main_metric = np.round(torch.mean(torch.Tensor(val_loss_list)).item(), 4)  
                              ##########################
                        #     if main_metric < best_metric:
                        #         best_metric = main_metric
                        #         model_filename = (f'm-{main_metric:.4f}.pkl')
                        #         model_path = args.save_dir + "/" + args.data_name + "-" + model_filename
                    ########################################################################################
                    ########################################################################################
                    loss_outputs["ce loss train"].append(train_loss_mean)
                    loss_outputs["ce loss valid"].append(valid_loss_mean)
                    loss_outputs["roc-auc"].append(roc_score)
                    loss_outputs["prc-auc"].append(prc_score)
                    loss_outputs["accuracy"].append(valid_accuracy)
                    ##########################
                    if args.is_visdom:
                        graph_title = args.data_name.upper()
                        line_name_t = "train" + "_" + str(cnt)
                        line_name_v = "valid" + "_" + str(cnt)
                        plotter.plot('CE Loss', line_name_t, graph_title, epoch_num, train_loss_mean)   # CE Loss T
                        plotter.plot('CE Loss', line_name_v, graph_title, epoch_num, valid_loss_mean)   # CE Loss V
                        plotter.plot('ROC-AUC', line_name_v, graph_title, epoch_num, roc_score)
                        plotter.plot('PRC-AUC', line_name_v, graph_title, epoch_num, prc_score)
                        plotter.plot('Accuracy', line_name_v, graph_title, epoch_num, valid_accuracy)   
                    ##########################
                    wandb.log({"CE Loss T": train_loss_mean, "CE Loss V": valid_loss_mean, "ROC-AUC": roc_score, "PRC-AUC": prc_score, "Accuracy": valid_accuracy})
                ########################################################################################
                ########################################################################################
    ##########################
    wandb.finish()
    ##########################
    loss_file_path = "../results/" + args.data_name + "/" + args.data_name + "_losses.json"
    with open(loss_file_path, "w") as f:
        json.dump(loss_outputs, f)
    ##########################
    # print(f"\n\n>>  Best ROC-AUC = % {max(rocs):.4f}")
    if args.is_cv:
        print(f"\n\n>>  {args.data_name.upper()} Training {args.tokenization}_{cnt} is COMPLETED.  <<\n")
    else:
        print(f"\n\n>>  {args.data_name.upper()} Training {args.tokenization} is COMPLETED.  <<\n")
    ########################################################################################
    ########################################################################################

########################################################################################
########################################################################################

def main(args):
    ########################################################################################
    temp_path = args.save_dir + "/" + args.data_name
    if os.path.exists(temp_path):   # klasör varsa
        shutil.rmtree(temp_path)   # klasör siliyor
    ##########################
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir, exist_ok=True)   # klasör oluşturuyor
    # else:
    #     pass
    os.mkdir(temp_path)
    ##########################
    frmt = '%(asctime)-30s %(levelname)-5s |  %(message)s'
    logging.basicConfig(level=logging.INFO, 
                        format=frmt,
                        datefmt='|  %Y-%m-%d  |  %H:%M:%S  |')
    logFormatter = logging.Formatter(frmt)
    ########################################################################################
    start = time.time()
    ##########################
    with open("cv_config.json", "r") as f:
        cv_all = json.load(f)
    cv_keys = list(cv_all.keys())
    cv_vals = list(cv_all.values())
    all_cv = list(itertools.product(*cv_vals))
    ##########################
    data = data_loaderX(args)
    ##########################
    print(f"\n\n>>  Constant Hyperparameters:\n")
    for k, v in vars(args).items():
        if k not in cv_keys:
            if k != "vocab":
                if k != "data_name":
                    logging.info(k + ' = ' + str(v))
    ##########################
    if args.is_cv:
        ##########################
        for cv_no in range(len(all_cv)):
            ##########################
            log_file_name = args.data_name + "/stdout" + "_" + args.data_name + ".log"
            rootLogger = logging.getLogger()
            fileHandler = logging.FileHandler(os.path.join(args.save_dir, log_file_name))
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)
            ##########################
            cv = all_cv[cv_no] 
            print(f"\n\n>>  {args.data_name.upper()} Training {args.tokenization}_{cv_no} STARTED.  <<")
            print(f"\n>>  Cross-validation Hyperparameters:\n")
            ##########################
            for param_no in range(len(cv_keys)):
                setattr(args, cv_keys[param_no], cv[param_no])
            ##########################
            for k, v in vars(args).items():
                if k == "data_name":
                    logging.info(k + ' = ' + str(v))
                if k in cv_keys:
                    logging.info(k + ' = ' + str(v))
            print("\n")
            ##########################
            train(args, cv_no, cv_keys, data)
        ##########################
        end = time.time()
        total = np.round(((end - start) / 60), 2)
        print(f"\n>>  {total} minutes elapsed for cross-validation.  <<\n")
        ##########################
    else:
        ##########################
        cv_no = 0
        log_file_name = args.data_name + "/stdout" + "_" + args.data_name + ".log"
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(args.save_dir, log_file_name))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        ##########################
        for k, v in vars(args).items():
            if k == "data_name":
                logging.info(k + ' = ' + str(v))
            if k in cv_keys:
                logging.info(k + ' = ' + str(v))
        print("\n")
        ##########################
        train(args, cv_no, cv_keys, data)
        ##########################
        end = time.time()
        total = np.round(((end - start) / 60), 2)
        print(f"\n>>  {total} minutes elapsed for the training.  <<\n")
        ##########################
    ########################################################################################

#######################################################################################
#######################################################################################

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenization', default='cha', choices=['bpe', 'cha'])
    parser.add_argument('--max_smi_len', default=100, type=int)
    parser.add_argument('--act_func', default="ReLU", type=str)
    parser.add_argument('--clf_num_layers', default=2, type=int)
    parser.add_argument('--wandb_mode', default="disabled", choices=["online", "offline", "disabled"], type=str)
    parser.add_argument('--is_cv', default=False)   # True
    parser.add_argument('--is_visdom', default=False)
    parser.add_argument('--is_scheduler', default=True)
    parser.add_argument('--is_clip', default=True)
    parser.add_argument('--data_name', required=True, type=str) 
    parser.add_argument('--save_dir', default='../results')
    parser.add_argument('--leaf_rnn_type', default='bilstm', choices=['bilstm', 'lstm'])
    parser.add_argument('--rank_input', default='w', choices=['w', 'h'], 
                        help='needed for STG, whether feed word embedding or hidden state of bilstm into score function')
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--tree_hidden_dim', default=300, type=int, 
                        help='dimension of final sentence embedding. each direction will be (hidden_dim // 2) when leaf rnn is bilstm')
    parser.add_argument('--DTA_hidden_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--use_batchnorm', default=True, action='store_true')
    parser.add_argument('--task', default='clf', choices=['clf'])   # , 'reg'
    ##########################
    parser.add_argument('--device', default="cuda", choices=['cuda', 'cpu'], type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)   # 50
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--clip', default=5.0, type=float)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adagrad', 'adadelta'])
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--mode', default='', help='for evaluation mode!')
    parser.add_argument('--fix_word_embedding', default=False, action='store_true')
    parser.add_argument('--is_pretrain', default="", type=str)
    args = parser.parse_args() 
    return args

########################################################################################
########################################################################################

if __name__ == '__main__':
    args = load_args()
    if args.is_visdom:
        from utils.plot_live_losses import TorchLossPlotter
        global plotter
        env_namex= "ART-Mol" + "_" + args.data_name.upper()
        plotter = TorchLossPlotter(env_name = env_namex)
    main(args)

########################################################################################
########################################################################################
########################################################################################




