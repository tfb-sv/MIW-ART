import argparse
import numpy as np
import torch
from torch import nn
from ete3 import Tree
from tqdm import tqdm
import pandas as pd
import math
import json
import pickle
import shutil
import sys
import os
from model.tree_model import ARTM_model, calc_metrics
from utils.tree_data import data_loader
torch.manual_seed(0)

def eval_iter(args, batch, model, criterion):
    model.eval()
    model_arg, labels, smis = batch
    probs, _ = model(**model_arg)
    threshold = 0.5
    if args.task == "clf": labels_pred = probs > threshold   
    elif args.task == "reg": pass
    labelz = torch.unsqueeze(labels.float(), dim=-1)
    loss = criterion(input=probs, target=labelz)
    if args.task == "clf":  
        labelsx = labels.cpu().detach().numpy().tolist()
        labels_predx = labels_pred.cpu().detach().numpy()
        labels_predx = labels_predx.astype("int")
        labels_predx = labels_predx.tolist()
        probsx = probs.cpu().detach().numpy().tolist()
        return loss, labelsx, labels_predx, probsx, smis
    elif args.task == "reg":
        labelsx = labelz.cpu().detach().numpy().tolist()
        labels_predx = probs.cpu().detach().numpy().tolist()
        return loss, labelsx, labels_predx, probs, smis

def postOrder(root):
    def recursion(node):
        if node is None: return "-"
        left = recursion(node.left)
        right = recursion(node.right)
        if node.left is None and node.right is None: return node.word   # leaf node
        else: return "(%s,%s)%s" % (left, right, node.word)
    return recursion(root)+";"

def getNewick(postOrderStr):
    t = Tree(postOrderStr, format=8)
    newick = t.write(format=8)
    return newick

def visualizeTree(postOrderStr):
    t = Tree(postOrderStr, format=8)
    print("\n", t.get_ascii())
    return

def main(args, data):
    if args.task == "clf": criterion = nn.BCELoss()
    elif args.task == "reg": criterion = nn.MSELoss()
    best_metric = 10 
    if args.mode in ["test"]:   
        modelX = ARTM_model
        model = modelX(args)
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device(args.device)))
        model.eval()
        model = model.to(args.device)
        labelZ, predZ, smileZ = [], [], []
        with tqdm(total=(data.num_test_batches), unit=" molecule", disable=args.tqdm_off) as pbar_test:
            test_loss_list, ground_truth, predictions, probabilities = [], [], [], []
            for test_batch_num, (test_batch) in enumerate(data.generator("test")):
                if args.task == "clf":
                    test_loss, labels, preds, probz, smis = eval_iter(args, test_batch, model, criterion)
                    ground_truth.extend(labels)
                    predictions.extend(preds)
                    probabilities.extend(probz)
                    pbar_test.set_description(f">>  MOLECULE {(test_batch_num + 1)}  |  BCE Loss = {test_loss.item():.4f}  |")
                elif args.task == "reg":
                    test_loss, labels, preds, prob,  smis = eval_iter(args, test_batch, model, criterion)
                    pbar_test.set_description(f">>  MOLECULE {(test_batch_num + 1)}  |  RMSE Loss = {test_loss.item():.4f}  |")
                labelZ.extend(labels)
                predZ.extend(preds)
                smileZ.extend(smis)
                test_loss_list.append(test_loss.item())
                pbar_test.update()
            if args.task == "clf": test_loss_mean = np.round(torch.mean(torch.Tensor(test_loss_list)).item(), 4)
            elif args.task == "reg": test_loss_mean = np.round(torch.sqrt(torch.mean(torch.Tensor(test_loss_list))).item(), 4)
        if args.task == "clf":
            roc_score, prc_score, test_accuracy = calc_metrics(ground_truth, predictions, probabilities, data)
            print(f"\n\n>>  {args.data_name.upper()} Testing is COMPLETED.  |  RESULTS:\n\n")
            print(f"|>>  BCE Loss % = {test_loss_mean}  \n|>>  ROC-AUC % = {roc_score}  \n|>>  PRC-AUC % = {prc_score} \n|>>  Accuracy % = {test_accuracy}\n\n")
        elif args.task == "reg":
            print(f"\n\n>>  {args.data_name.upper()} Testing is COMPLETED.  |  RESULTS:\n\n")
            print(f"|>>  RMSE Loss = {test_loss_mean}\n\n")
    elif args.mode in ["newick", "visualize"]:
        task_path = f"{args.eval_save_dir}/{args.data_name}"
        if not os.path.exists(args.eval_save_dir): os.mkdir(args.eval_save_dir)
        if os.path.exists(task_path): shutil.rmtree(task_path)
        os.mkdir(task_path)
        modelX = ARTM_model
        model = modelX(args)
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device(args.device)))
        model.eval()
        model = model.to(args.device)
        vis_decoder_chem_dict = data.word_to_id_l
        with tqdm(total=(data.num_all_batches), unit=" molecule", disable=args.tqdm_off) as pbar_test:
            all_newicks = {}
            for test_batch_num, (test_batch) in enumerate(data.generator("all")):
                if args.task == "clf": test_loss, _, _, _, _ = eval_iter(args, test_batch, model, criterion)
                elif args.task == "reg": test_loss, _, _, _, _ = eval_iter(args, test_batch, model, criterion)
                model_arg = test_batch[0]
                label = test_batch[1].item()
                smi = test_batch[2][0]
                _, supplements = model(**model_arg)
                if args.mode == "newick":
                    newick = getNewick(postOrder(supplements["tree"][0]))
                    all_newicks[smi] = [newick, test_loss.item(), label]
                elif args.mode == "visualize":
                    visualizeTree(postOrder(supplements["tree"][0]))
                    print(f"\n{smi}")
                pbar_test.set_description(f">>  MOLECULE {(test_batch_num + 1)}  |")
                pbar_test.update()
        newicks_save_path = f"{args.eval_save_dir}/{args.data_name}/all_newicks_{args.data_name}.json"
        with open(newicks_save_path, "w") as f: json.dump(all_newicks, f)
        if args.mode == "newick": print(f"\n\n>>  {args.data_name.upper()} Newicking is COMPLETED.  <<\n")
        elif args.mode == "visualize": print(f"\n\n>>  {args.data_name.upper()} Visualizing is COMPLETED.  <<\n")

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqdm_off", default=False, type=bool)
    parser.add_argument("--x_label", default="smiles", type=str)
    parser.add_argument("--y_label", default="y_true", type=str)
    parser.add_argument("--data_folder", default="../data", type=str)
    parser.add_argument("--mode", default="test", choices=["test", "newick", "visualize"])
    parser.add_argument("--data_name", default="")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--eval_load_dir", default="../results/training_results")
    parser.add_argument("--eval_save_dir", default="../results/evaluation_results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", default=1, type=int)   
    parser.add_argument("--task", default="clf", choices=["clf", "reg"])
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
    parser.add_argument("--DTA_hidden_dim", default=512, type=int)
    parser.add_argument("--clf_num_layers", default=2, type=int)
    args = parser.parse_args() 
    return args

if __name__ == "__main__":
    args = load_args()
    print(f"\n")
    subfile_path = f"{args.eval_load_dir}/{args.data_name}"
    with open("utils/best_hyprs.json", "r") as f: best_hyprs = json.load(f)
    task_type = best_hyprs[args.data_name]["task"][0]
    all_pkls = {}
    for subfile in os.listdir(subfile_path):
        if subfile.endswith(".pkl"):
            tmp_score = subfile[:-4].split("-")[3]
            all_pkls[float(tmp_score)] = subfile
    if not all_pkls: 
        print("\n\n>>  !  ERROR  !  NO .PKL FILE FOUND  !  <<\n\n")
        exit(0)
    if task_type == "clf": subfile = all_pkls[max(all_pkls.keys())]
    elif task_type == "reg": subfile = all_pkls[min(all_pkls.keys())]
    print(f"\n>>  {args.data_name.upper()}  |\n")
    args_file_path = f"{subfile_path}/m-args-{args.data_name}-0.json"
    with open(args_file_path, "r") as f: model_args = json.load(f)
    for key in model_args:
        if key == "batch_size": continue
        if key in list(vars(args).keys()):
            if key == "mode": continue
            setattr(args, key, model_args[key])
    ckpt_path = f"{subfile_path}/{subfile}"
    args.ckpt = ckpt_path
    print(f"\n>>  {args.data_name.upper()} {subfile} {args.mode.upper()} is STARTED.  <<")
    data = data_loader(args)
    main(args, data)

    
  