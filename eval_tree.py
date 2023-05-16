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
import shutil
torch.manual_seed(0)

########################################################################################

def eval_iter(args, batch, model, criterion):
    #######################################################################################
    # eps = 1e-8
    model.eval()   # .train(False) ???
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
    # print(f"\npredcs={probs}\nlabels{labels}\n")
    labelz = torch.unsqueeze(labels.float(), dim=-1)
    if args.mode == "emb":
        probsx = probs.cpu().detach().numpy().tolist()
        return labelz, [], [], probsx, []
    loss = criterion(input=probs, target=labelz)
    ####################################################
    if args.task == "clf":  
        labelsx = labels.cpu().detach().numpy().tolist()   # NRL
        labels_predx = labels_pred.cpu().detach().numpy()
        labels_predx = labels_predx.astype("int")
        labels_predx = labels_predx.tolist()   # NRL
        probsx = probs.cpu().detach().numpy().tolist()
        return loss, labelsx, labels_predx, probsx, smis
    ##########################
    elif args.task == "reg":
        # loss = torch.sqrt(loss + eps)   # for RMSE
        labelsx = labelz.cpu().detach().numpy().tolist()
        labels_predx = probs.cpu().detach().numpy().tolist()
        return loss, labelsx, labels_predx, probs, smis
    ####################################################
    #######################################################################################

########################################################################################

def legal(s):
    s2 = s
    # s2 = s.replace("[", "!")
    # s2 = s2.replace("]", "?")
    # s2 = s2.replace("=", "*")
    return s2

########################################################################################

def postOrder(root):
    def recursion(node):
        if node is None:
            return "-"
        left = recursion(node.left)
        right = recursion(node.right)
        new_token = legal(node.word)
        if node.left is None and node.right is None:
            return new_token   # leaf node
        else:
            return "(%s,%s)%s" % (left, right, new_token)
    return recursion(root)+";"

########################################################################################

def getNewick(postOrderStr):
    t = Tree(postOrderStr, format=8)
    newick = t.write(format=8)
    if "_" in newick:
        print(newick)
    # elif "?" in newick:
    #     print(newick)
    return newick

########################################################################################
    
def visualizeTree(postOrderStr):
    t = Tree(postOrderStr, format=8)
    print("\n", t.get_ascii())
    return

########################################################################################

def main(args, hyp_no, data):
    ########################################################################################
    if args.task == "clf":
        criterion = nn.BCELoss()   # nn.CrossEntropyLoss()
    elif args.task == "reg":
        criterion = nn.MSELoss()
    best_metric = 10
    #######################################################################################################################
    #######################################################################################################################   
    if args.mode in ["test", "emb"]:   
        ##########################
        rocs, prcs, accs, ces = [], [], [], []
        ####################################################
        ####################################################
        modelX = ARTM_model
        model = modelX(args)
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device(args.device)))   # loaded ???
        model.eval()   # .train(True) ???
        model = model.to(args.device)
        ##########################
        labelZ, predZ, smileZ = [], [], []
        ##########################
        if args.mode == "emb":
            real_labels = []
        ##########################
        with tqdm(total=(data.num_test_batches), unit=" molecule", disable=args.tqdm_off) as pbar_test:
            test_loss_list, ground_truth, predictions, probabilities = [], [], [], []
            ##########################
            for test_batch_num, (test_batch) in enumerate(data.generator("test")):
                ##########################
                if args.task == "clf":
                    test_loss, labels, preds, probz, smis = eval_iter(args, test_batch, model, criterion)
                    ground_truth.extend(labels)
                    predictions.extend(preds)
                    pbar_test.set_description(f">>  MOLECULE {(test_batch_num + 1)}  |  BCE Loss = {test_loss.item():.4f}  |")
                ##########################
                elif args.task == "reg":
                    test_loss, labels, preds, prob,  smis = eval_iter(args, test_batch, model, criterion)
                    pbar_test.set_description(f">>  MOLECULE {(test_batch_num + 1)}  |  RMSE Loss = {test_loss.item():.4f}  |")
                    # test_loss = torch.square(test_loss)
                ##########################
                labelZ.extend(labels)
                predZ.extend(preds)
                smileZ.extend(smis)
                test_loss_list.append(test_loss.item())
                ##########################
                if args.mode == "emb":
                    real_label = test_batch[1].item()
                    real_labels.append(real_label)
                    probabilities.append(probz)
                else:
                    probabilities.extend(probz)
                ##########################
                pbar_test.update()
                ####################################################
            ####################################################
            if args.mode == "emb":
                embZ = pd.DataFrame(probabilities)
                # print("\n", embZ.columns, "\n")
                real_labels = pd.DataFrame(real_labels)
                embZ = pd.concat([real_labels, embZ], axis=1)
                embZ.columns = ["PubChem_CID", "Embedding"]
                # embZ.to_csv(f"../results/embedding_{args.data_name.upper()}.csv")
                embZ.to_pickle(f"../results/embedding_{args.data_name.upper()}.pkl")
                return
            ####################################################
            if args.export_preds:
                ##########################
                predZ = pd.DataFrame(predZ)
                ##########################
                labelZ = pd.DataFrame(labelZ)
                ##########################
                smileZ = pd.DataFrame(smileZ)
                ##########################
                fdf = pd.concat([smileZ, labelZ, predZ], axis=1)
                fdf.columns = ["smiles", "y_true", "y_pred"]
                ##########################
                fdf.to_csv(f"../results/ys_check_{args.data_name.upper()}.csv")
                ##########################
            ####################################################
            ####################################################
            # deal_with_embs(xx)
            ####################################################
            if args.task == "clf":
                test_loss_mean = np.round(torch.mean(torch.Tensor(test_loss_list)).item(), 4)
            ##########################
            elif args.task == "reg":
                test_loss_mean = np.round(torch.sqrt(torch.mean(torch.Tensor(test_loss_list))).item(), 4)
        ####################################################
        if args.task == "clf":
            roc_score, prc_score, test_accuracy = calc_metrics(ground_truth, predictions, probabilities, data)
            rocs.append(roc_score)
            prcs.append(prc_score)
            accs.append(test_accuracy)
            ces.append(test_loss_mean)
            #################################################### BEGINNING OF THE POINTLESS SECTION !!!!!!!!!!!!!!!!!!!!!!!!!!!
            roc_score = np.round((np.mean(rocs) * 100), 1)
            prc_score = np.round((np.mean(prcs) * 100), 1)
            test_accuracy = np.round((np.mean(accs) * 100), 1)
            test_loss_mean = np.round((np.mean(ces) * 100), 1)
            ##########################
            roc_std = np.round((np.std(rocs) * 100), 1)
            prc_std = np.round((np.std(prcs) * 100), 1)
            acc_std = np.round((np.std(accs) * 100), 1)
            ce_std = np.round((np.std(ces) * 100), 1)
            ##########################
            print(f"\n\n>>  {args.data_name.upper()} {args.tokenization} Testing is COMPLETED.  |  RESULTS:\n\n")
            print(f"|>>  BCE Loss % = {test_loss_mean} ({ce_std})  \n|>>  ROC-AUC % = {roc_score} ({roc_std})  \n|>>  PRC-AUC % = {prc_score} ({prc_std})  \n|>>  Accuracy % = {test_accuracy} ({acc_std})\n\n")
        ####################################################
        elif args.task == "reg":
            ##########################
            print(f"\n\n>>  {args.data_name.upper()} {args.tokenization} Testing is COMPLETED.  |  RESULTS:\n\n")
            print(f"|>>  RMSE Loss = {test_loss_mean}\n\n")
            ##########################
        ######################################################################################################## END OF THE POINTLESS SECTION, CLEAR HERE. !!!!!!!!!!!!!!!!!!!!!!!!!!!
        ########################################################################################################
        if args.export_preds:
            ##########################
            smileZ = pd.DataFrame(smileZ)
            ##########################
            labelZ = pd.DataFrame(labelZ)
            ##########################
            predZ = pd.DataFrame(predZ)
            ##########################
            fdf = pd.concat([smileZ, labelZ, predZ], axis=1)
            fdf.columns = ["smiles", "y_true", "y_pred"]
            ##########################
            fdf.to_csv(f"../results/ys_check_{args.data_name.upper()}.csv")
            ##########################
        ####################################################
        ####################################################
    #######################################################################################################################
    #######################################################################################################################
    elif args.mode in ["newick", "visualize"]:
        ####################################################
        task_path = (f"{args.eval_save_dir}/{args.data_name}")
        ##########################
        if not os.path.exists(args.eval_save_dir):
            os.mkdir(args.eval_save_dir, exist_ok=True)
        ##########################
        if os.path.exists(task_path):
            shutil.rmtree(task_path)
        #########################
        os.mkdir(task_path)
        ####################################################
        modelX = ARTM_model
        model = modelX(args)
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device(args.device)))   # loaded ???
        model.eval()   # .train(True) ???
        model = model.to(args.device)
        ##########################
        vis_decoder_chem_dict = data.word_to_id_l
        ####################################################
        with tqdm(total=(data.num_all_batches), unit=" molecule", disable=args.tqdm_off) as pbar_test:
            all_newicks = {}
            ####################################################
            for test_batch_num, (test_batch) in enumerate(data.generator("all")):
                ##########################
                if args.task == "clf":
                    test_loss, _, _, _, _ = eval_iter(args, test_batch, model, criterion)
                elif args.task == "reg":
                    test_loss, _, _, _, _ = eval_iter(args, test_batch, model, criterion)
                ##########################
                model_arg = test_batch[0]
                label = test_batch[1].item()
                smi = test_batch[2][0]
                logits, supplements = model(**model_arg)   # logits gereksiz
                ##########################
                if args.mode == "newick":
                    newick = getNewick(postOrder(supplements["tree"][0]))
                    all_newicks[smi] = [newick, test_loss.item(), label]
                elif args.mode == "visualize":
                    visualizeTree(postOrder(supplements["tree"][0]))
                    print(f"\n{smi}")
                ##########################
                pbar_test.set_description(f">>  MOLECULE {(test_batch_num + 1)}  |")
                pbar_test.update()
        ####################################################
        newicks_save_path = (f"{args.eval_save_dir}/{args.data_name}/all_newicks_{args.data_name}.json")
        with open(newicks_save_path, "w") as f:
            json.dump(all_newicks, f)
        ##########################
        print(f"\n\n>>  {args.data_name.upper()} {args.tokenization} Newicking is COMPLETED.  <<\n")
    #######################################################################################################################    

########################################################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    ##########################
    parser.add_argument("--export_preds", default=False, type=bool)
    parser.add_argument("--tqdm_off", default=False, type=bool)
    parser.add_argument("--data_names", default="", type=str)
    parser.add_argument("--x_label", default="smiles", type=str)
    parser.add_argument("--y_label", default="y_true", type=str)   # y_true
    parser.add_argument("--data_folder", default="data", type=str)
    parser.add_argument("--is_debug", default=False, action="store_true")
    parser.add_argument("--mode", default="test", choices=["test", "newick", "emb", "visualize"])
    parser.add_argument("--data_name", default="")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--eval_load_dir", default="../results/training_results")
    parser.add_argument("--eval_save_dir", default="../results/evaluation_results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tokenization", default="cha", choices=["bpe", "cha"])
    parser.add_argument("--batch_size", default=1, type=int)   
    parser.add_argument("--task", default="clf", choices=["clf", "reg"])
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
    parser.add_argument("--DTA_hidden_dim", default=512, type=int)
    parser.add_argument("--clf_num_layers", default=2, type=int)
    ##########################
    args = parser.parse_args() 
    ##########################
    return args

########################################################################################

if __name__ == "__main__":
    ########################################################################################
    args = load_args()
    ##########################
    print(f"\n")
    ##########################
    data_names = args.data_names.replace(" ", "")
    data_names = data_names.split(",")
    ########################################################################################
    # for task_name in os.listdir(args.eval_load_dir):   # THIS IS AN ALL_in_ONE PROCEDURE !
    for task_name in data_names:
        if "." in task_name:
            continue
        if "saveds" in task_name:
            continue
        args.data_name = task_name
        ##########################
        print(f"\n>>  {task_name.upper()}  |\n")
        ##########################
        subfile_path = (f"{args.eval_load_dir}/{task_name}")
        for subfile in os.listdir(subfile_path):
            if subfile.endswith(".pkl"):
                hyp_no = subfile[:-4].split("-")[4]
                args_file_path = (f"{subfile_path}/m-args-{args.data_name}-{hyp_no}.json")
                ##########################
                with open(args_file_path, "r") as f:
                    model_args = json.load(f)
                for key in model_args:
                    if key == "batch_size":
                        continue
                    if key in list(vars(args).keys()):
                        # if args.mode == "emb":   # FOR GKC_emb
                            # if key in ["x_label", "y_label", "data_name", "mode"]:
                            #     continue
                        if key == "mode":
                            continue
                        setattr(args, key, model_args[key])
                ##########################
                ckpt_path = (f"{subfile_path}/{subfile}")
                args.ckpt = ckpt_path
                ####################################################
                if args.mode == "newick":   # düzeltilmesi gerekiyor kaydedilen dosya için ya da hiç for döngüsünde olmayacak ???????????
                    if args.ckpt == "":
                        print("\n\n>>  !  ERROR  !  NO CKPT FILE FOUND FOR TESTING  !  <<\n\n")
                        break
                    print(f"\n>>  {args.data_name.upper()} {subfile} {args.tokenization} Newicking STARTED.  <<")
                    data = data_loaderX(args)
                ####################################################
                elif args.mode == "test":
                    if args.ckpt == "":
                        print("\n\n>>  !  ERROR  !  NO CKPT FILE FOUND FOR TESTING  !  <<\n\n")
                        break
                    print(f"\n\n>>  {args.data_name.upper()} {subfile} {args.tokenization} Testing STARTED.  <<")
                    data = data_loaderX(args)
                ####################################################
                elif args.mode == "emb":
                    if args.ckpt == "":
                        print("\n\n>>  !  ERROR  !  NO CKPT FILE FOUND FOR TESTING  !  <<\n\n")
                        break
                    print(f"\n\n>>  {args.data_name.upper()} {subfile} {args.tokenization} Embedding STARTED.  <<")
                    data = data_loaderX(args)
                ####################################################
                elif args.mode == "visualize":
                    if args.ckpt == "":
                        print("\n\n>>  !  ERROR  !  NO CKPT FILE FOUND FOR TESTING  !  <<\n\n")
                        break
                    print(f"\n\n>>  {args.data_name.upper()} {subfile} {args.tokenization} Visualization STARTED.  <<")
                    data = data_loaderX(args)
                ####################################################
                args.tqdm_off = False
                main(args, hyp_no, data)
    ########################################################################################
            
########################################################################################















































