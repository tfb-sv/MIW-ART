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
from model.SingleModel import SingleModel
from dataLoader import TheData
from evaluate import eval_iter
from tqdm import tqdm
import sys
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from plot_losses import TorchLossPlotter
# import matplotlib.pyplot as plt
torch.manual_seed(0)

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
CUDA_VISIBLE_DEVICES=1

def train_iter(args, batch, model, params, criterion, optimizer):
    model.train(True)
    model_arg, labels = batch
    logits, _ = model(**model_arg)
    if args.task == "clf":
        labels_pred = logits.max(1)[1]
        num_correct = torch.eq(labels, labels_pred).long().sum().item()       
    elif args.task == "reg":
        logits = logits.view(-1)
    loss = criterion(input=logits, target=labels)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(parameters=params, max_norm=args.clip)
    optimizer.step()
    if args.task == "clf":
        return loss, num_correct, labels.cpu().detach().numpy().tolist(), labels_pred.cpu().detach().numpy().tolist()
    elif args.task == "reg":
        return loss

def train_rl_iter(args, batch, model, params, criterion, optimizer):
    model.train(True)
    model_arg, labels = batch
    sample_num = args.sample_num
    logits, supplements = model(**model_arg)
    if args.task == "clf":
        labels_pred = logits.max(1)[1]
        num_correct = torch.eq(labels, labels_pred).long().sum().item()       
    elif args.task == "reg":
        logits = logits.view(-1)
    sv_loss = criterion(input=logits, target=labels)
    ######################################################################################
    #rl training loss for sampled trees
    sample_logits, probs, sample_trees = supplements['sample_logits'], supplements['probs'], supplements['sample_trees']
    sample_label_pred = sample_logits.max(1)[1]
    sample_label_gt = labels.unsqueeze(1).expand(-1, sample_num).contiguous().view(-1)
    rl_rewards = torch.eq(sample_label_gt, sample_label_pred).float().detach() * 2 - 1 #NRL#NRL#NRL#NRL#NRL
    rl_loss = 0
    #average of word
    final_probs = defaultdict(list)
    for i in range(len(labels)):
        cand_rewards = rl_rewards[i*sample_num: (i+1)*sample_num]
        for j in range(sample_num):
            k = i * sample_num + j
            for w in probs[k]:
                final_probs[w] += [p*rl_rewards[k] for p in probs[k][w]]
    for w in final_probs:
        rl_loss += - sum(final_probs[w]) / len(final_probs[w])
    if len(final_probs) > 0:
        rl_loss /= len(final_probs)
    rl_loss *= args.rl_weight
    ######################################################################################
    total_loss = sv_loss + rl_loss
    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm_(parameters=params, max_norm=args.clip)
    optimizer.step()
    if args.task == "clf":
        return total_loss, rl_loss, num_correct, labels.cpu().detach().numpy().tolist(), labels_pred.cpu().detach().numpy().tolist()
    elif args.task == "reg":
        return total_loss, rl_loss

def train(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = device
    data = TheData(args)
    num_train_batches = data.num_train_batches
    num_valid_batches = data.num_valid_batches
    Model = SingleModel
    model_kwargs = { k:v for k,v in vars(args).items() if k in
                   { 'model_type', 'leaf_rnn_type', 'rank_input', 'word_dim', 'tree_hidden_dim', 
                     'DTA_hidden_dim', 'dropout', 'use_batchnorm', 'task', 'mode', 'tokenization', 
                     'molecule', 'df_loc', "cv_cnt"}}
    model = Model(**vars(args))
    # if args.pre_train != "":
        # model.load_state_dict(torch.load(args.pre_train), strict=False)   # strict, 1e1 eşleşmezse hata vermesini engelliyor, isim eşleşmesine bakıyor
        # for name, param in model.named_parameters():
            # if name.startswith("classifier."):
                # param.requires_grad = False   # güncellememesi için bu layerları !!
    if args.fix_word_embedding:
        logging.info('* Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    model = model.to(device)
    logging.info(model)
    params = [p for p in model.parameters() if p.requires_grad]   
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    else:
        raise Exception('unknown optimizer')    
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=args.patience, verbose=True)
    if args.task == "clf":
        criterion = nn.CrossEntropyLoss()
        tag = "CE"
    elif args.task == "reg":
        criterion = nn.MSELoss()
        tag = "MSE"
    trpack = [model, params, criterion, optimizer]
    best_val_loss = 0
    print("\n")   
    ########################################################################################
    for epoch_num in range(args.max_epoch):
        train_loss_list = []
        train_rl_loss_list = []
        val_loss_list = []
        rocs = []
        prcs = []
        with tqdm(total=(num_train_batches), unit="batch") as pbar_train:                      
            total_correct = 0
            predictions = []
            ground_truth = []
            for batch_iter, (train_batch) in enumerate(data.train_minibatch_generator()):
                if args.model_type == 'RL':
                    if args.task == "clf":
                        train_loss, train_rl_loss, curr_correct, labels, preds = train_rl_iter(args, train_batch, *trpack)
                        train_rl_loss_list.append(train_rl_loss.item())
                        total_correct += curr_correct
                        predictions.extend(preds)
                        ground_truth.extend(labels)
                    elif args.task == "reg":
                        train_loss, train_rl_loss = train_rl_iter(args, train_batch, *trpack) #torch.Tensor([0.25]) #
                        train_rl_loss_list.append(train_rl_loss.item())
                elif args.model_type == 'STG':
                    if args.task == "clf":
                        train_loss, curr_correct, labels, preds = train_iter(args, train_batch, *trpack) #train_loss, accuracy
                        total_correct += curr_correct
                        predictions.extend(preds)
                        ground_truth.extend(labels)
                    elif args.task == "reg":
                        train_loss = train_iter(args, train_batch, *trpack) #torch.Tensor([0.25]) #                                  
                train_loss_list.append(train_loss.item())   
                pbar_train.set_description(f'    |  CELoss: {train_loss.item():.4f}  |  Epoch {epoch_num+1}  |')
                pbar_train.update()
                ########################################################################################
                if (batch_iter + 1) % num_train_batches == 0:                  
                    train_loss_mean = np.round(torch.mean(torch.Tensor(train_loss_list)).item(), 4)
                    if args.model_type == "RL":
                        train_rl_loss_mean = torch.mean(torch.Tensor(train_rl_loss_list)).item()
                    if args.task == "clf":              
                        train_accuracy = (total_correct / data.train_size)*100
                        roc_score = np.round(roc_auc_score(ground_truth, predictions), 4)
                    elif args.task == "reg":
                        pass
                    with tqdm(total=(num_valid_batches), unit="batch") as pbar_val:
                        valid_batch_counter = 0
                        total_correct = 0
                        predictions = []
                        ground_truth = []
                        for valid_batch in data.valid_minibatch_generator():                           
                            valid_batch_counter += 1
                            if args.task == "clf":
                                _, val_loss, curr_correct, labels, preds = eval_iter(valid_batch, model, args.task, args.mode) #torch.Tensor([0.3]) #
                                total_correct += curr_correct
                                predictions.extend(preds)
                                ground_truth.extend(labels)
                            elif args.task == "reg": 
                                _, val_loss = eval_iter(valid_batch, model, args.task) #torch.Tensor([0.3]) #
                            val_loss_list.append(val_loss.item())
                        roc_score = np.round(roc_auc_score(ground_truth, predictions), 4)
                        precision, recall, _ = precision_recall_curve(ground_truth, predictions)
                        prc_score = np.round(auc(recall, precision), 4)
                        prcs.append(prc_score)
                        rocs.append(roc_score)
                        if args.task == "reg": 
                            val_loss_mean = np.round(torch.mean(torch.Tensor(val_loss_list)).item(), 4)
                        elif args.task == "clf":
                            val_loss_mean = roc_score                       
                        ########################################################################################
                        is_checkpoint = False                      
                        if args.task == "clf":
                            valid_accuracy = (total_correct / data.valid_size) * 100
                            val_ce_mean = np.round(torch.mean(torch.Tensor(valid_loss_list)).item(), 4)
                            if val_loss_mean > best_val_loss:   # val_loss_mean artık roc_score
                                best_val_loss = val_loss_mean   # best_val_loss da artık roc_score lool :)
                                model_filename = (f'm-{val_loss_mean:.4f}.pkl')
                                model_path = args.save_dir + "/" + args.df_loc[5:] + "-" + model_filename
                                is_checkpoint = True
                                pbar_val.set_description(f'    |  MODEL SAVED!  |  ROC-AUC: % {roc_score:.4f}  |  PRC-AUC: % {prc_score:.4f}  |  Epoch {epoch_num+1}  |')
                                pbar_val.update()
                            else:
                                pbar_val.set_description(f'    |  ROC-AUC: % {roc_score:.4f}  |  PRC-AUC: % {prc_score:.4f}  |  Epoch {epoch_num+1}  |')
                                pbar_val.update()
                        elif args.task == "reg":
                            if val_loss_mean < best_val_loss:   # regression için < olmalı NRL NURAAAAL 
                                best_val_loss = val_loss_mean
                                model_filename = (f'm-{val_loss_mean:.4f}.pkl')
                                model_path = args.save_dir + "/" + args.df_loc[5:] + "-" + model_filename
                                is_checkpoint = True
                        if is_checkpoint:
                            save_checkpoint(model, model_kwargs, model_path)
                        ########################################################################################
                        graph_title = args.df_loc[5:].upper()
                        line_name_t = "Train" + "_" + str(args.cv_cnt)
                        line_name_v = "Val" + "_" + str(args.cv_cnt)
                        plotter.plot('CELoss', line_name_t, graph_title, epoch_num, train_loss_mean)
                        plotter.plot('ROC', line_name_v, graph_title, epoch_num, val_loss_mean)   #roc-auc score, not ce!
                        plotter.plot('CELoss', line_name_v, graph_title, epoch_num, val_ce_mean)
                        plotter.plot('Acc', line_name_v, graph_title, epoch_num, valid_accuracy)
                        ########################################################################################
                        print("\n")
    print(f"\n    |  TRAINING HAS BEEN COMPLETED!  |  Best Val ROC-AUC = % {max(rocs):.4f}  |   Best Val PRC-AUC = % {max(prcs):.4f}  |\n")
    ########################################################################################

def save_checkpoint(model, model_kwargs, path):
    state = {'model': model.state_dict(),
             'model_kwargs': model_kwargs}
    # print("Model is checkpointed.")
    torch.save(state, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_cnt', default=0, type=int)
    parser.add_argument('--df_loc', required=True) 
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--model-type', default='STG', choices=['RL', 'STG'])
    parser.add_argument('--leaf-rnn-type', default='bilstm', choices=['bilstm', 'lstm'])
    parser.add_argument('--rank-input', default='w', choices=['w', 'h'], help='whether feed word embedding or hidden state of bilstm into score function')
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--tree-hidden-dim', default=300, type=int, help='dimension of final sentence embedding. each direction will be hidden_dim//2 when leaf rnn is bilstm')
    parser.add_argument('--DTA-hidden-dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--tokenization', default='bpe', choices=['bpe', 'cha'], help='byte-pair encoding or character-based')
    parser.add_argument('--use-batchnorm', default=True, action='store_true')
    parser.add_argument('--task', default='clf', choices=['clf', 'reg'], help='classification or regression')
    ##########################
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--sample-num', default=3, type=int, help='sample num for reinforce')
    # parser.add_argument('--pre-train', default="pre_train/pretrain-m-1.8696.pkl")
    parser.add_argument('--rl_weight', default=0.1, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--max-epoch', default=25, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adagrad', 'adadelta'])
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--mode', default='', help='not important but do not delete this argument')
    parser.add_argument('--fix-word-embedding', default=False, action='store_true')
    parser.add_argument('--molecule', default='ligand', choices=['ligand'], help='artik gerek bile yok sanirim da neyse..')
    #parser.add_argument('--clf-num-layers', type=int, default=3)
    args = parser.parse_args()  
    ########################################################################################
    # a simple log file, the same content as stdout
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    ########################################################################################
    df_loc_orj = copy.deepcopy(str(args.df_loc))
    start = time.time()
    args.cv_cnt = 0
    for a1 in [64, 32]:
        for a2 in [300, 500]:
            for a3 in [0.3, 0.5]:
                for a4 in ["bpe", "cha"]:
                    args.batch_size = a1
                    args.tree_hidden_dim = a2
                    args.dropout = a3
                    args.tokenization = a4
                    for k, v in vars(args).items():
                        logging.info(k+':'+str(v))
                    for i in range(3):   # for file_no in [122, 123, 124]:                   
                        file_loc = "data/" + str(df_loc_orj)   #  + str(file_no)
                        print("\n>>", file_loc, "is started to training procedure.\n")
                        args.df_loc = file_loc
                        train(args)
                    args.cv_cnt += 1
    end = time.time()
    total = np.round(((end - start) / 60), 2)
    print("\n>>", total, "minutes passed.")

if __name__ == '__main__':
    global plotter
    plotter = TorchLossPlotter(env_name = 'ARTM')
    main()





