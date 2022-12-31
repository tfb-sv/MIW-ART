########################################################################################

import argparse
import numpy as np
import torch
from torch import nn
from model.tree_model import ARTM_model, calc_metrics
from tree_data import data_loaderX
from ete3 import Tree
from tqdm import tqdm
import pandas as pd
import math
import json
import pickle
import os
torch.manual_seed(0)

########################################################################################

def invert_dict(d):
    return { v:k for k,v in d.items() }

########################################################################################

def eval_iter(args, batch, model):
    # mode = args.mode
    model.eval()   # .train(False) ???
    if args.task == "clf":
        model_arg = batch[0]
        labels = batch[1]
        # smis = batch[2]
        # if mode == "emb":
        #     logits = model(**model_arg)   # actually not logits, but hyp_h matrix
        #     return logits
    # elif args.task == "reg":
    #     model_arg, labels = batch
    logits, _ = model(**model_arg)
    if args.task == "clf":
        labels_pred = logits.max(1)[1]
        num_correct = torch.eq(labels, labels_pred).long().sum().item()  
        criterion = nn.CrossEntropyLoss()
        loss = criterion(input=logits, target=labels)
        labelsx = labels.cpu().detach().numpy().tolist()   # NRL
        labels_predx = labels_pred.cpu().detach().numpy().tolist()   # NRL
        return logits, loss, num_correct, labelsx, labels_predx
    # elif args.task == "reg":
    #     logits = logits.view(-1)
    #     criterion = nn.MSELoss()
    #     loss = criterion(input=logits, target=labels)
    #     return logits, loss

########################################################################################

def legal(s):
    return s.replace(",", "<comma>")

########################################################################################

def postOrder(root):
    def recursion(node):
        if node is None:
            return "-"
        left = recursion(node.left)
        right = recursion(node.right)
        if node.left is None and node.right is None:
            return legal(node.word) # leaf node
        else:
            return "(%s,%s)%s" % (left, right, legal(node.word))
    return recursion(root)+";"

########################################################################################

def getNewick(postOrderStr):
    t = Tree(postOrderStr, format=8)
    newick = t.write(format=8)
    return newick

########################################################################################

def main(args):
    tokenization = args.tokenization
    data = data_loaderX(args)   
    modelX = ARTM_model
    model = modelX(args)
    loaded = torch.load(args.ckpt, map_location=torch.device(args.device))
    model.load_state_dict(loaded)
    model.eval()   # .train(True) ???
    model = model.to(args.device)
    #######################################################################################################################
    #######################################################################################################################   
    if args.mode == "test":   
        ##########################
        with tqdm(total=(data.num_test_batches), unit=" molecule") as pbar_test:
            total_correct = 0
            test_loss_list, predictions, ground_truth = [], [], []
            ##########################
            for test_batch_num, (test_batch) in enumerate(data.generator("test")):
                ##########################
                if args.task == "clf":
                    _, test_loss, curr_correct, labels, preds = eval_iter(args, test_batch, model)
                    total_correct += curr_correct
                    predictions.extend(preds)
                    ground_truth.extend(labels)
                ##########################
                # elif args.task == "reg":
                #     _, test_loss = eval_iter(args, test_batch, *trpack)
                ##########################
                pbar_test.set_description(f">>  MOLECULE {(test_batch_num + 1)}  |  CE Loss = {test_loss.item():.4f}  |")
                pbar_test.update()
                test_loss_list.append(test_loss.item())
        ##########################
        test_loss_mean = np.round(torch.mean(torch.Tensor(test_loss_list)).item(), 4)
        ##########################
        if args.task == "clf":
            roc_score, prc_score, test_accuracy = calc_metrics(ground_truth, predictions, total_correct, data)
            ##########################
            print(f"\n\n>>  {args.data_name.upper()} {args.tokenization} Training is COMPLETED.  |  ROC-AUC = {roc_score:.4f}  |  CE Loss = {test_loss_mean:.4f}  |\n")
    #######################################################################################################################
    #######################################################################################################################
    elif args.mode == "newick":
        ##########################
        if tokenization == "bpe":
            with open("utils/vocabs/chemical/chembl27_bpe_32000.json", "r") as f1:
                vis_decoder_chem_dict = json.load(f1)
            vis_decoder_chem_dict = invert_dict(vis_decoder_chem_dict["model"]["vocab"])
        ##########################
        elif tokenization == "cha":
            with open("data/INV_CHARSET.json", "r") as f:
                vis_decoder_chem_dict = json.load(f) 
        ##########################
        cnt = 0
        ##########################
        with tqdm(total=(data.num_test_batches), unit=" molecule") as pbar_test:
            all_newicks = {}
            ##########################
            for test_batch_num, (test_batch) in enumerate(data.generator("test")):
                model_arg = test_batch[0]
                # labels = test_batch[1]
                smi = test_batch[2]
                logits, supplements = model(**model_arg)   # logits gereksiz
                newick = getNewick(postOrder(supplements["tree"][0]))
                all_newicks[list(smi)[0]] = newick
                ##########################
                pbar_test.update()
                cnt += 1
            ##########################  
        ##########################
        newicks_save_path = args.save_dir + "/all_newicks_" + args.data_name + ".json"
        with open(newicks_save_path, "w") as f:
            json.dump(all_newicks, f)
        ##########################
        print(f"\n\n>>  {args.data_name.upper()} {args.tokenization} Newicking is COMPLETED.  <<\n")
    #######################################################################################################################
    #######################################################################################################################      

########################################################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="newick", choices=["test", "newick"])
    parser.add_argument("--data_name", default="")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--save_dir", default="../results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tokenization", default="cha", choices=["bpe", "cha"])
    parser.add_argument("--batch_size", default=1, type=int)   
    parser.add_argument("--task", default="clf", choices=["clf"])   # , "reg"
    ##########################
    parser.add_argument("--max_smi_len", default=100, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--word_dim", default=300, type=int)
    parser.add_argument("--leaf_rnn_type", default="bilstm", choices=["bilstm", "lstm"])
    parser.add_argument("--tree_hidden_dim", default=300, type=int, 
                        help="dimension of final sentence embedding. each direction will be (hidden_dim // 2) when leaf rnn is bilstm")
    parser.add_argument("--rank_input", default="w", choices=["w", "h"], 
                        help="needed for STG, whether feed word embedding or hidden state of bilstm into score function")
    parser.add_argument("--act_func", default="ReLU", type=str)
    parser.add_argument("--use_batchnorm", default=True, action="store_true")
    parser.add_argument("--DTA_hidden_dim", default=1024, type=int)
    parser.add_argument("--clf_num_layers", default=2, type=int)
    ##########################
    args = parser.parse_args() 
    ##########################
    return args

########################################################################################

if __name__ == "__main__":
    ########################################################################################
    args = load_args()
    ########################################################################################
    for task_name in os.listdir(args.save_dir):   # THIS IS AN ALL_in_ONE PROCEDURE !
        if "." in task_name:
            continue
        if task_name == "saveds":
            continue
        args.data_name = task_name
        subfile_path = args.save_dir + "/" + task_name
        for subfile in os.listdir(subfile_path):
            if subfile.endswith(".pkl"):
                ckpt_path = subfile_path + "/" + subfile
                args.ckpt = ckpt_path
        if args.mode == "newick":
            print(f"\n\n>>  {args.data_name.upper()} Newicking {args.tokenization} STARTED.  <<")
        elif args.mode == "test":
            if args.ckpt == "":
                print("\n\n>>  !  ERROR  !  NO CKPT FILE FOUND FOR TESTING  !  <<\n\n")
                break
            print(f"\n\n>>  {args.data_name.upper()} Testing {args.tokenization} STARTED.  <<")
        main(args)
    ########################################################################################
        
        
########################################################################################
