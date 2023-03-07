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
from model.tree_model import ARTM_model, calc_metrics
from tree_data import data_loaderX
from eval_tree import eval_iter
from tqdm import tqdm
import sys
import json
import numpy as np
import copy
import wandb
import itertools
torch.manual_seed(0)
CUDA_VISIBLE_DEVICES=1

########################################################################################
########################################################################################

def train_iter(args, batch, model, params, criterion, optimizer):
    #######################################################################################
    eps = 1e-8
    model.train(True)
    ##########################
    model_arg, labels, smis = batch
    ##########################
    probs, _ = model(**model_arg)
    threshold = 0.5
    ####################################################
    if args.task == "clf":
        labels_pred = probs > threshold      
    ##########################
    elif args.task == "reg":
        pass
    ####################################################
    labelz = torch.unsqueeze(labels.float(), dim=-1)
    loss = criterion(input=probs, target=labelz)
    loss = torch.sqrt(loss + eps)
    ##########################
    optimizer.zero_grad()
    loss.backward()
    ##########################
    if args.is_clip:   # NRL
        clip_grad_norm_(parameters=params, max_norm=args.clip) 
    ##########################
    optimizer.step()
    ####################################################
    if args.task == "clf":
        labelsx = labels.cpu().detach().numpy().tolist()
        labels_predx = labels_pred.cpu().detach().numpy().tolist()
        probsx = probs.cpu().detach().numpy().tolist()
        return loss, labelsx, labels_predx, probsx
    ##########################
    elif args.task == "reg":
        return loss
    ####################################################
    #######################################################################################

########################################################################################
########################################################################################

def train(args, cnt, cv_keyz, data, key):
    ######################################################################################## INITIALIZE WANDB
    if args.is_debug:
        args.wandb_mode = "disabled"   # "online"   #
        args.max_epoch = 2
        project_name = (f"try_project")
    else:
        project_name = (f"{args.proj_name}_{args.data_name.upper()}")
    ##########################
    if args.is_cv == "ideal":
        run_name = (f"hypc_{cnt}")
        prmz = {cv_keyz[i]: getattr(args, cv_keyz[i]) for i in range(len(cv_keyz))}
    elif args.is_cv == "feasible":
        run_name = (f"{key}_{getattr(args, key)}_{cnt}")
        prmz = {key: getattr(args, key)}
    elif args.is_cv == "besty":
        run_name = (f"besty_{cnt}")
        prmz = {cv_keyz[i]: getattr(args, cv_keyz[i]) for i in range(len(cv_keyz))}
    ##########################
    if args.wandb_mode == "online":
        token_note = args.tokenization
        run = ""
        while run == "":
            try:
                run = wandb.init(mode=args.wandb_mode, 
                                 project=project_name, 
                                 config=prmz, 
                                 name=run_name, 
                                 notes=token_note, 
                                 reinit=True, 
                                 force=True, 
                                 settings=wandb.Settings(start_method='thread'))
            except:
                pass
    ######################################################################################## BUILD MODEL
    modelX = ARTM_model
    model = modelX(args)
    ######################################################################################## SETUP MODEL
    # if args.is_pretrain != "":
        # model.load_state_dict(torch.load(args.pretrain_file), strict=False)   # strict, 1e1 eşleşmezse hata vermesini engelliyor, isim eşleşmesine bakıyor !!!
        # for name, param in model.named_parameters():
            # if name.startswith("classifier."):
                # param.requires_grad = False
    ##########################
    if args.fix_word_embedding:   # NRL
        model.word_embedding.weight.requires_grad = False 
    ##########################
    model = model.to(args.device)
    ##########################
    logging.info(model)
    print("\n")
    ######################################################################################## SETUP OPTIMIZER AND SCHEDULER
    params = [p for p in model.parameters() if p.requires_grad]  
    ##########################
    if args.optimizer == "adam":
        optimizer_class = optim.Adam
    elif args.optimizer == "adagrad":
        optimizer_class = optim.Adagrad
    elif args.optimizer == "adadelta":
        optimizer_class = optim.Adadelta
    ##########################
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)
    ##########################
    if args.is_scheduler:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.5, patience=args.patience, verbose=True) 
    ######################################################################################## SET LOSS FUNCTION
    if args.task == "clf":
        criterion = nn.BCELoss()   # nn.CrossEntropyLoss()
        ##########################
        loss_outputs = {
                        "bce loss train": [],
                        "bce loss valid": [],
                        "roc-auc": [],
                        "prc-auc": [],
                        "accuracy": [],
                        "bce loss test": [],
                        "roc-aucT": [],
                        "prc-aucT": [],
                        "accuracyT": []
                        }
    ##########################
    elif args.task == "reg":
        criterion = nn.MSELoss()
        ##########################
        loss_outputs = {
                        "rmse loss train": [],
                        "rmse loss valid": [],
                        "rmse loss test": []
                        }
    ##########################
    best_metric = 10
    second_best_metric = 0
    ######################################################################################## SET DATA HOLDERS
    trpack = [model, params, criterion, optimizer]
    ######################################################################################## START TRAIN LOOPS
    ########################################################################################
    for epoch_num in range(args.max_epoch):
        train_loss_list, val_loss_list = [], []
        with tqdm(total=(data.num_train_batches), unit="batch") as pbar_train:                      
            for train_batch_num, (train_batch) in enumerate(data.generator("train")):
                if args.task == "clf":
                    train_loss, _, _, _ = train_iter(args, train_batch, *trpack)
                elif args.task == "reg":
                    train_loss = train_iter(args, train_batch, *trpack) 
                ##########################
                train_loss_list.append(train_loss.item())   
                pbar_train.set_description(f">>  EPOCH {(epoch_num + 1)}T  |  RMSE Loss = {train_loss.item():.4f}  |")
                pbar_train.update()
                ######################################################################################## START VALID LOOP
                ########################################################################################
                if (train_batch_num + 1) % data.num_train_batches == 0:  
                    train_loss_mean = np.round(torch.mean(torch.Tensor(train_loss_list)).item(), 4)
                    pbar_train.set_description(f">>  EPOCH {(epoch_num + 1)}T  |  RMSE Loss = {train_loss_mean:.4f}  |")
                    pbar_train.update()
                    with tqdm(total=(data.num_valid_batches), unit="batch") as pbar_val:
                        ground_truth, predictions, probabilities = [], [], []
                        ########################################################################################
                        for valid_batch_num, (valid_batch) in enumerate(data.generator("valid")):                           
                            if args.task == "clf":
                                val_loss, labels, preds, probz = eval_iter(args, valid_batch, model, criterion)
                                ground_truth.extend(labels)
                                predictions.extend(preds)
                                probabilities.extend(probz)
                            ##########################
                            elif args.task == "reg": 
                                val_loss = eval_iter(args, valid_batch, model, criterion)
                            val_loss_list.append(val_loss.item())
                            pbar_val.update()
                        ########################################################################################   # NRL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        ground_truthT, predictionsT, probabilitiesT, test_loss_list = [], [], [], []
                        for test_batch_num, (test_batch) in enumerate(data.generator("test")):
                            ##########################
                            if args.task == "clf":
                                test_loss, labels, preds, probz = eval_iter(args, test_batch, model, criterion)
                                ground_truthT.extend(labels)
                                predictionsT.extend(preds)
                                probabilitiesT.extend(probz)
                                ########################## NRL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                roc_scoreT, prc_scoreT, test_accuracy = calc_metrics(ground_truthT, predictionsT, probabilitiesT, data)
                            ##########################
                            elif args.task == "reg":
                                test_loss = eval_iter(args, test_batch, model, criterion)
                            ##########################
                            test_loss_list.append(test_loss.item()) 
                        ######################################################################################## CALCULATE COMMON METRICS
                        test_loss_mean = np.round(torch.mean(torch.Tensor(test_loss_list)).item(), 4)
                        valid_loss_mean = np.round(torch.mean(torch.Tensor(val_loss_list)).item(), 4)
                        ######################################################################################## PROCESSES FOR CLF             
                        if args.task == "clf":
                            roc_score, prc_score, valid_accuracy = calc_metrics(ground_truth, predictions, probabilities, data)
                            ##########################
                            main_metric = test_loss_mean   # valid   # NRL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            second_metric = roc_scoreT   # roc_score   # NRL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            ####################################################
                            if second_metric > second_best_metric:   # roc_score
                                second_best_metric = second_metric
                                ##########################
                                save_model(args, model, "roc", second_best_metric, cnt)
                            ####################################################
                            if main_metric < best_metric:   # bce_loss
                                best_metric = main_metric
                                ##########################
                                save_model(args, model, "bce", best_metric, cnt)
                                ##########################
                                pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  MODEL SAVED.  |  RMSE Loss = {valid_loss_mean}  |")
                                pbar_val.update()
                            ####################################################
                            else:
                                pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  RMSE Loss = {valid_loss_mean}  |")
                                pbar_val.update()
                        ######################################################################################## PROCESSES FOR REG  
                        elif args.task == "reg": 
                            main_metric = test_loss_mean  
                            ##########################
                            if main_metric < best_metric:
                                save_model(args, model, "rmse", best_metric, cnt)
                                ##########################
                                pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  MODEL SAVED.  |  RMSE Loss = {valid_loss_mean}  |")
                                pbar_val.update()
                            ####################################################
                            else:
                                pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  RMSE Loss = {valid_loss_mean}  |")
                                pbar_val.update()
                    ######################################################################################## EXPORT DATA TO HOLDERS
                    ########################################################################################
                    if args.task == "clf":
                        loss_outputs["bce loss train"].append(train_loss_mean)
                        loss_outputs["bce loss valid"].append(valid_loss_mean)
                        loss_outputs["roc-auc"].append(roc_score)
                        loss_outputs["prc-auc"].append(prc_score)
                        loss_outputs["accuracy"].append(valid_accuracy)
                        ##########################   # NRL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        loss_outputs["bce loss test"].append(test_loss_mean)
                        loss_outputs["roc-aucT"].append(roc_scoreT)
                        loss_outputs["prc-aucT"].append(prc_scoreT)
                        loss_outputs["accuracyT"].append(test_accuracy)
                        ##########################
                        # if args.is_visdom:
                            # graph_title = args.data_name.upper()
                            # line_name_t = (f"train_{cnt}")
                            # line_name_v = (f"valid_{cnt}")
                            # plotter.plot("CE Loss", line_name_t, graph_title, epoch_num, train_loss_mean)   # CE Loss T
                            # plotter.plot("CE Loss", line_name_v, graph_title, epoch_num, valid_loss_mean)   # CE Loss V
                            # plotter.plot("ROC-AUC", line_name_v, graph_title, epoch_num, roc_score)
                            # plotter.plot("PRC-AUC", line_name_v, graph_title, epoch_num, prc_score)
                            # plotter.plot("Accuracy", line_name_v, graph_title, epoch_num, valid_accuracy)   
                        ##########################
                        if args.wandb_mode == "online":
                            wandb.log({"BCE Loss T": train_loss_mean, "BCE Loss V": valid_loss_mean, "ROC-AUC": roc_score, "PRC-AUC": prc_score, "Accuracy": valid_accuracy, "BCE Loss Test": test_loss_mean, "ROC-AUCT": roc_scoreT, "PRC-AUCT": prc_scoreT, "AccuracyT": test_accuracy})
                    ########################################################################################
                    if args.task == "reg":
                        loss_outputs["rmse loss train"].append(train_loss_mean)
                        loss_outputs["rmse loss valid"].append(valid_loss_mean)
                        ##########################   # NRL !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        loss_outputs["rmse loss test"].append(test_loss_mean)
                        ##########################
                        # if args.is_visdom:
                            # graph_title = args.data_name.upper()
                            # line_name_t = (f"train_{cnt}")
                            # line_name_v = (f"valid_{cnt}")
                            # plotter.plot("BCE Loss", line_name_t, graph_title, epoch_num, train_loss_mean)   # BCE Loss T
                            # plotter.plot("BCE Loss", line_name_v, graph_title, epoch_num, valid_loss_mean)   # BCE Loss V 
                        ##########################
                        if args.wandb_mode == "online":
                            wandb.log({"RMSE Loss T": train_loss_mean, "RMSE Loss V": valid_loss_mean, "RMSE Loss Test": test_loss_mean})
                    ########################################################################################
                ########################################################################################
                ######################################################################################## FINISH EVERYTHING
    ##########################
    if args.wandb_mode == "online":
        run.finish()
    ##########################
    loss_file_path = (f"{args.train_save_dir}/{args.data_name}/{args.data_name}_metrics_{cnt}.json")
    with open(loss_file_path, "w") as f:
        json.dump(loss_outputs, f)
    ##########################
    if args.is_cv == "ideal":
        print(f"\n\n>>  {args.data_name.upper()} {args.tokenization}_{cnt} Training is COMPLETED.  <<\n")
    elif args.is_cv == "feasible":
        print(f"\n\n>>  {args.data_name.upper()} {key}_{cnt} Training is COMPLETED.  <<\n")
    elif args.is_cv == "besty":
        print(f"\n\n>>  {args.data_name.upper()} {args.tokenization} Training is COMPLETED.  <<\n")
    ########################################################################################
    ######################################################################################## END OF THE STORY, THANK YOU

########################################################################################
########################################################################################

def save_model(args, model, metric, best_metric, cnt):
    ####################################################
    model_filename = (f"{metric.upper()}-m-{args.data_name}-{best_metric:.4f}-{cnt}.pkl")
    model_path = (f"{args.train_save_dir}/{args.data_name}/{model_filename}")
    ##########################
    torch.save(model.state_dict(), model_path)
    ####################################################
    model_args_filename = (f"m-args-{args.data_name}-{cnt}.json")
    model_args_path = (f"{args.train_save_dir}/{args.data_name}/{model_args_filename}")
    ##########################
    model_args = {}
    for k, v in vars(args).items():
        if k in ["vocab"]:
            continue
        model_args[k] = v
    ##########################
    with open(model_args_path, "w") as f:
        json.dump(model_args, f)
    ####################################################
    return
    
########################################################################################
########################################################################################

def main(args):
    ########################################################################################
    temp_path = (f"{args.train_save_dir}/{args.data_name}")
    if os.path.exists(temp_path):   # klasör varsa, training_results/task
        shutil.rmtree(temp_path)   # klasör siliyor, training_results/task
    ##########################
    if not os.path.exists(args.train_save_dir):
        os.mkdir(args.train_save_dir, exist_ok=True)   # klasör oluşturuyor, training_results
    os.mkdir(temp_path)   # klasör oluşturuyor, training_results/task
    ##########################
    frmt = "%(asctime)-30s %(levelname)-5s |  %(message)s"
    logging.basicConfig(level=logging.INFO, 
                        format=frmt,
                        datefmt="|  %Y-%m-%d  |  %H:%M:%S  |")
    logFormatter = logging.Formatter(frmt)
    ########################################################################################
    start = time.time()
    ##########################
    with open("cv_config.json", "r") as f:
        cv_all = json.load(f)
    cv_keys = list(cv_all[args.data_name].keys())
    ##########################
    print(f">>  Constant Hyperparameters:\n")
    for k, v in vars(args).items():
        if k not in cv_keys:
            if k not in ["vocab", "data_name"]:
                logging.info(k + " = " + str(v))
    ########################################################################################
    if args.is_cv == "feasible":
        ##########################
        for key in cv_keys:
            if len(cv_all[args.data_name][key]) == 1:
                setattr(args, key, cv_all[args.data_name][key][0])
                logging.info(key + " = " + str(cv_all[args.data_name][key][0]))
        ##########################
        cv_no = 0
        for key in cv_keys:
            ##########################
            orj_key_value = getattr(args, key)
            if len(cv_all[args.data_name][key]) == 1:
                continue
            ##########################
            for hyp in cv_all[args.data_name][key]:
                log_file_name = (f"{args.data_name}/stdout_{args.data_name}.log")
                rootLogger = logging.getLogger()
                fileHandler = logging.FileHandler(os.path.join(args.train_save_dir, log_file_name))
                fileHandler.setFormatter(logFormatter)
                rootLogger.addHandler(fileHandler)
                ##########################
                print(f"\n\n>>  {args.data_name.upper()} {key} = {hyp} (hypc_{cv_no}) Training STARTED.  <<")
                print(f"\n>>  Cross-validation Hyperparameters:\n")
                ##########################
                setattr(args, key, hyp)
                ##########################
                data = data_loaderX(args)
                ##########################
                for k, v in vars(args).items():
                    if k == "data_name":
                        logging.info(k + " = " + str(v))
                    if k in cv_keys:
                        if len(cv_all[args.data_name][k]) != 1:
                            logging.info(k + " = " + str(v))
                ##########################
                train(args, cv_no, cv_keys, data, key)
                cv_no += 1
            ##########################
            setattr(args, key, orj_key_value)
        ##########################
        end = time.time()
        total = np.round(((end - start) / 60), 2)  
        print(f"\n>>  {total} minutes elapsed for cross-validation.  <<\n")
        ##########################
        # ########################################################################################
    elif args.is_cv == "ideal":
        ##########################
        cv_vals = list(cv_all[args.data_name].values())
        all_cv = list(itertools.product(*cv_vals))
        key = ""
        ##########################
        for cv_no in range(len(all_cv)):
            log_file_name = (f"{args.data_name}/stdout_{args.data_name}.log")
            rootLogger = logging.getLogger()
            fileHandler = logging.FileHandler(os.path.join(args.train_save_dir, log_file_name))
            fileHandler.setFormatter(logFormatter)
            rootLogger.addHandler(fileHandler)
            ##########################
            cv = all_cv[cv_no] 
            print(f"\n\n>>  {args.data_name.upper()} hypc_{cv_no} Training STARTED.  <<")
            print(f"\n>>  Cross-validation Hyperparameters:\n")
            ##########################
            for param_no in range(len(cv_keys)):
                setattr(args, cv_keys[param_no], cv[param_no])
            ##########################
            data = data_loaderX(args)
            ##########################
            for k, v in vars(args).items():
                if k == "data_name":
                    logging.info(k + " = " + str(v))
                if k in cv_keys:
                    logging.info(k + " = " + str(v))
            ##########################
            train(args, cv_no, cv_keys, data, key)
        ##########################
        end = time.time()
        total = np.round(((end - start) / 60), 2)
        print(f"\n>>  {total} minutes elapsed for cross-validation.  <<\n")
        ##########################
    # ########################################################################################
    elif args.is_cv == "besty":
        ##########################
        for key in cv_keys:
            setattr(args, key, cv_all[args.data_name][key][0])
        ##########################
        cv_no = 0
        log_file_name = (f"{args.data_name}/stdout_{args.data_name}.log")
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(args.train_save_dir, log_file_name))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        ##########################
        print(f"\n\n>>  {args.data_name.upper()} {args.tokenization} Training STARTED.  <<")
        ##########################
        data = data_loaderX(args)
        key = ""
        ##########################
        for k, v in vars(args).items():
            if k == "data_name":
                logging.info(k + " = " + str(v))
            if k in cv_keys:
                logging.info(k + " = " + str(v))
        ##########################
        for i in range(args.init_repeat):
            train(args, cv_no, cv_keys, data, key)
            cv_no += 1
        ##########################
        end = time.time()
        total = np.round(((end - start) / 60), 2)
        print(f"\n>>  {total} minutes elapsed for the training.  <<\n")
        ##########################
    ########################################################################################

#######################################################################################
#######################################################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", default="ART_MolinT7", type=str)
    parser.add_argument("--init_repeat", default=1, type=int)
    parser.add_argument("--is_debug", default=False, action="store_true")
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"], type=str)
    parser.add_argument("--is_cv", default="feasible", choices=["ideal", "feasible", "besty"], type=str)
    parser.add_argument("--max_epoch", default=150, type=int)
    parser.add_argument("--tokenization", default="cha", choices=["bpe", "cha"])
    parser.add_argument("--max_smi_len", default=100, type=int)
    parser.add_argument("--act_func", default="ReLU", type=str)
    parser.add_argument("--clf_num_layers", default=1, type=int)
    parser.add_argument("--is_visdom", default=False)
    parser.add_argument("--is_scheduler", default=True)
    parser.add_argument("--is_clip", default=True)
    parser.add_argument("--data_name", required=True, type=str) 
    parser.add_argument("--train_save_dir", default="../results/training_results")
    parser.add_argument("--leaf_rnn_type", default="bilstm", choices=["bilstm", "lstm"])
    parser.add_argument("--rank_input", default="w", choices=["w", "h"], 
                        help="needed for STG, whether feed word embedding or hidden state of bilstm into score function")
    parser.add_argument("--word_dim", default=300, type=int)
    parser.add_argument("--tree_hidden_dim", default=300, type=int, 
                        help="dimension of final sentence embedding. each direction will be (hidden_dim // 2) when leaf rnn is bilstm")
    parser.add_argument("--DTA_hidden_dim", default=512, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--use_batchnorm", default=True, action="store_true")
    parser.add_argument("--task", default="clf", choices=["clf", "reg"])
    # parser.add_argument("--mode", default="", help="for evaluation mode!")
    ##########################
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--l2reg", default=1e-5, type=float)
    parser.add_argument("--clip", default=5.0, type=float)
    parser.add_argument("--optimizer", default="adadelta", choices=["adam", "adagrad", "adadelta"])
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--fix_word_embedding", default=False, action="store_true")
    parser.add_argument("--is_pretrain", default="", type=str)
    # parser.add_argument("--pretrain_file", default="", type=str)
    ##########################
    args = parser.parse_args()
    ##########################    
    return args

########################################################################################
########################################################################################

if __name__ == "__main__":
    args = load_args()
    # if args.is_visdom:
        # from utils.plot_live_losses import TorchLossPlotter
        # global plotter
        # env_namex= (f"ART-Mol_{args.data_name.upper()}")
        # plotter = TorchLossPlotter(env_name = env_namex)
    if "OneDrive" in os.getcwd():
        args.is_debug = True
    else:
        args.is_debug = False
    main(args)

########################################################################################
########################################################################################
########################################################################################




