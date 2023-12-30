import argparse
import logging
import time
import shutil
from collections import defaultdict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import softmax
from tqdm import tqdm
import json
import numpy as np
import copy
import itertools
import sys
import os
from model.tree_model import ARTM_model, calc_metrics
from utils.tree_data import data_loader
from eval_tree import eval_iter
torch.manual_seed(0)
CUDA_VISIBLE_DEVICES=1

def train_iter(args, batch, model, params, criterion, optimizer):
    model.train(True)
    model_arg, labels, smis = batch
    probs, _ = model(**model_arg)
    threshold = 0.5
    if args.task == "clf": labels_pred = probs > threshold      
    elif args.task == "reg": pass
    labelz = torch.unsqueeze(labels.float(), dim=-1)
    loss = criterion(input=probs, target=labelz)
    optimizer.zero_grad()
    loss.backward()
    if args.is_clip: clip_grad_norm_(parameters=params, max_norm=args.clip) 
    optimizer.step()
    if args.task == "clf":
        labelsx = labels.cpu().detach().numpy().tolist()
        labels_predx = labels_pred.cpu().detach().numpy().tolist()
        probsx = probs.cpu().detach().numpy().tolist()
        return loss, labelsx, labels_predx, probsx
    elif args.task == "reg": return loss
        
def train(args, cnt, data):
    # BUILD MODEL
    modelX = ARTM_model
    model = modelX(args)
    # SETUP MODEL
    if args.fix_word_embedding: model.word_embedding.weight.requires_grad = False 
    model = model.to(args.device)
    logging.info(model)
    print("\n")
    # SETUP OPTIMIZER AND SCHEDULER
    params = [p for p in model.parameters() if p.requires_grad]  
    if args.optimizer == "adam": optimizer_class = optim.Adam
    elif args.optimizer == "adagrad": optimizer_class = optim.Adagrad
    elif args.optimizer == "adadelta": optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)
    if args.is_scheduler: scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.5, patience=args.patience, verbose=True) 
    # SET LOSS FUNCTION
    if args.task == "clf":
        criterion = nn.BCELoss()
        loss_outputs = {
                        "bce loss train": [],
                        "bce loss valid": [],
                        "roc-auc": [],
                        "prc-auc": [],
                        "accuracy": []
                        }
    elif args.task == "reg":
        criterion = nn.MSELoss()
        loss_outputs = {
                        "rmse loss train": [],
                        "rmse loss valid": []
                        }
    best_metric = 10
    second_best_metric = 0
    # SET DATA HOLDERS
    trpack = [model, params, criterion, optimizer]
    for epoch_num in range(args.max_epoch):
        train_loss_list, valid_loss_list = [], []
        with tqdm(total=(data.num_train_batches), unit="batch", disable=args.tqdm_off) as pbar_train:                      
            for train_batch_num, (train_batch) in enumerate(data.generator("train")):
                if args.task == "clf":
                    train_loss, _, _, _ = train_iter(args, train_batch, *trpack)
                    pbar_train.set_description(f">>  EPOCH {(epoch_num + 1)}T  |  BCE Loss = {train_loss.item():.4f}  |")
                elif args.task == "reg":
                    train_loss = train_iter(args, train_batch, *trpack) 
                    pbar_train.set_description(f">>  EPOCH {(epoch_num + 1)}T  |  RMSE Loss = {train_loss.item():.4f}  |")
                train_loss_list.append(train_loss.item())   
                pbar_train.update()
                if (train_batch_num + 1) % data.num_train_batches == 0: 
                    if args.task == "clf":
                        train_loss_mean = np.round(torch.mean(torch.Tensor(train_loss_list)).item(), 4)
                        pbar_train.set_description(f">>  EPOCH {(epoch_num + 1)}T  |  BCE Loss = {train_loss_mean:.4f}  |")
                    elif args.task == "reg":
                        train_loss_mean = np.round(torch.sqrt(torch.mean(torch.Tensor(train_loss_list))).item(), 4)
                        pbar_train.set_description(f">>  EPOCH {(epoch_num + 1)}T  |  RMSE Loss = {train_loss_mean:.4f}  |")
                    pbar_train.update()
                    with tqdm(total=(data.num_valid_batches), unit="batch", disable=args.tqdm_off) as pbar_val:
                        ground_truth, predictions, probabilities = [], [], []
                        for valid_batch_num, (valid_batch) in enumerate(data.generator("valid")):                           
                            if args.task == "clf":
                                valid_loss, labels, preds, probz, _ = eval_iter(args, valid_batch, model, criterion)
                                ground_truth.extend(labels)
                                predictions.extend(preds)
                                probabilities.extend(probz)
                            elif args.task == "reg": valid_loss, _, _, _, _ = eval_iter(args, valid_batch, model, criterion)
                            valid_loss_list.append(valid_loss.item())
                            pbar_val.update()
                        # CALCULATE COMMON METRICS
                        if args.task == "clf": valid_loss_mean = np.round(torch.mean(torch.Tensor(valid_loss_list)).item(), 4)
                        elif args.task == "reg": valid_loss_mean = np.round(torch.sqrt(torch.mean(torch.Tensor(valid_loss_list))).item(), 4)
                        # PROCESSES FOR CLF             
                        if args.task == "clf":
                            roc_score, prc_score, valid_accuracy = calc_metrics(ground_truth, predictions, probabilities, data)
                            main_metric = valid_loss_mean
                            second_metric = roc_score
                            if second_metric > second_best_metric:
                                second_best_metric = second_metric
                                # save_model(args, model, "roc", second_best_metric, cnt)
                            if main_metric < best_metric:
                                best_metric = main_metric
                                save_model(args, model, "bce", best_metric, cnt)
                                pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  MODEL SAVED.  |  BCE Loss = {valid_loss_mean}  |")
                            else: pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  BCE Loss = {valid_loss_mean}  |")
                            pbar_val.update()
                        # PROCESSES FOR REG  
                        elif args.task == "reg": 
                            main_metric = valid_loss_mean  
                            if main_metric < best_metric:
                                best_metric = main_metric
                                save_model(args, model, "rmse", best_metric, cnt)
                                pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  MODEL SAVED.  |  RMSE Loss = {valid_loss_mean}  |")
                            else: pbar_val.set_description(f">>  EPOCH {(epoch_num + 1)}V  |  RMSE Loss = {valid_loss_mean}  |")
                            pbar_val.update()
                    # EXPORT OUTPUT TO HOLDERS
                    if args.task == "clf":
                        loss_outputs["bce loss train"].append(train_loss_mean)
                        loss_outputs["bce loss valid"].append(valid_loss_mean)
                        loss_outputs["roc-auc"].append(roc_score)
                        loss_outputs["prc-auc"].append(prc_score)
                        loss_outputs["accuracy"].append(valid_accuracy)
                    if args.task == "reg":
                        loss_outputs["rmse loss train"].append(train_loss_mean)
                        loss_outputs["rmse loss valid"].append(valid_loss_mean)
    loss_file_path = f"{args.train_save_dir}/{args.data_name}/{args.data_name}_metrics_{cnt}.json"
    with open(loss_file_path, "w") as f: json.dump(loss_outputs, f)
    print(f"\n\n>>  {args.data_name.upper()} Training is COMPLETED.  <<\n")

def save_model(args, model, metric, best_metric, cnt):
    model_filename = f"{metric.upper()}-m-{args.data_name}-0.0001-{cnt}.pkl"   # {best_metric:.4f}
    model_path = f"{args.train_save_dir}/{args.data_name}/{model_filename}"
    torch.save(model.state_dict(), model_path)
    model_args_filename = f"m-args-{args.data_name}-{cnt}.json"
    model_args_path = f"{args.train_save_dir}/{args.data_name}/{model_args_filename}"
    model_args = {}
    for k, v in vars(args).items():
        if k in ["vocab"]: continue
        model_args[k] = v
    with open(model_args_path, "w") as f: json.dump(model_args, f)
    return

def main(args):
    task_path = f"{args.train_save_dir}/{args.data_name}"
    if not os.path.exists(args.train_save_dir): os.mkdir(args.train_save_dir)
    if os.path.exists(task_path): shutil.rmtree(task_path)
    os.mkdir(task_path)
    frmt = "%(asctime)-30s %(levelname)-5s |  %(message)s"
    logging.basicConfig(level=logging.INFO, 
                        format=frmt,
                        datefmt="|  %Y-%m-%d  |  %H:%M:%S  |")
    logFormatter = logging.Formatter(frmt)
    start = time.time()
    with open("utils/best_hyprs.json", "r") as f: best_hyprs = json.load(f)
    hypr_keys = list(best_hyprs[args.data_name].keys())
    print(f">>  Constant Hyperparameters:\n")
    for k, v in vars(args).items():
        if k not in hypr_keys:
            if k not in ["vocab", "data_name"]: logging.info(k + " = " + str(v))
    for key in hypr_keys: setattr(args, key, best_hyprs[args.data_name][key][0])
    cv_no = 0
    log_file_name = f"{args.data_name}/stdout_{args.data_name}.log"
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.train_save_dir, log_file_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    print(f"\n\n>>  {args.data_name.upper()} Training is STARTED.  <<")
    for k, v in vars(args).items():
        if k == "data_name": logging.info(k + " = " + str(v))
        if k in hypr_keys: logging.info(k + " = " + str(v))
    data = data_loader(args)
    train(args, cv_no, data)
    end = time.time()
    total = np.round(((end - start) / 60), 2)
    print(f"\n>>  {total} minutes elapsed for the training.  <<\n")

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqdm_off", default=False, type=bool)
    parser.add_argument("--x_label", default="smiles", type=str)
    parser.add_argument("--y_label", default="y_true", type=str)
    parser.add_argument("--data_folder", default="../data", type=str)
    parser.add_argument("--max_epoch", default=150, type=int)
    parser.add_argument("--max_smi_len", default=100, type=int)
    parser.add_argument("--act_func", default="ReLU", type=str)
    parser.add_argument("--clf_num_layers", default=1, type=int)
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
    parser.add_argument("--mode", default="", help="for the evaluation of model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--l2reg", default=1e-5, type=float)
    parser.add_argument("--clip", default=5.0, type=float)
    parser.add_argument("--optimizer", default="adadelta", choices=["adam", "adagrad", "adadelta"])
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--fix_word_embedding", default=False, action="store_true")
    args = parser.parse_args() 
    return args

if __name__ == "__main__":
    args = load_args()
    main(args)
   