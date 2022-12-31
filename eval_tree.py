########################################################################################

import argparse
import numpy as np
import torch
from torch import nn
from model.tree_model import ARTM_model
from tree_data import data_loaderX
from ete3 import Tree
from tqdm import tqdm
import pandas as pd
import math
import json
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
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
    return s.replace(',', '<comma>')

########################################################################################

def ids2words(ids, decoder_dict, tokenization):
    sentence = []
    for i in range(len(ids)):
        if tokenization == "bpe":
            word = decoder_dict[ids[i]]
        elif tokenization == "cha":
            word = decoder_dict[str(ids[i])]
        sentence.append(word)
    return sentence

########################################################################################

def postOrder(root):
    def recursion(node):
        if node is None:
            return '-'
        left = recursion(node.left)
        right = recursion(node.right)
        if node.left is None and node.right is None:
            return legal(node.word) # leaf node
        else:
            return '(%s,%s)%s' % (left, right, legal(node.word))
    return recursion(root)+';'

########################################################################################

def visualizeTree(postOrderStr):
    t = Tree(postOrderStr, format=8)
    t_ascii = t.get_ascii(show_internal=True)
    print(t_ascii)

########################################################################################
    
def gradeNodes(postOrderStr, freq_dict, smi, test_loss):
    t = Tree(postOrderStr, format=8)
    newick = str(t.write())
    sn = [smi, newick]
    root = t.get_tree_root()
    _, max_point = root.get_farthest_leaf()
    max_point += 1
    for node in t.traverse("levelorder"):
        token = node.name
        if token == "-":
            continue
        point = max_point - t.get_distance(root, node)       
        if token not in freq_dict:   # token hiç sözlükte yoksa
            freq_dict[token] = [[sn, max_point, test_loss, [point]]]
        else:   # token sözlükte varsa
            truth = 0
            for temp_lst in freq_dict[token]:
                prev_sn = temp_lst[0]
                if sn == prev_sn:   # token aynı kimyasalda birden fazla varsa
                    truth = 1
                    point_lst = temp_lst[3]
                    point_lst.append(point)
                    temp_lst[3] = point_lst
                    break
            if truth == 0:   # token bu kimyasalda ilk kez bulunuyorsa
                temp = freq_dict[token]
                temp.append([sn, max_point, test_loss, [point]])
                freq_dict[token] = temp               
    return freq_dict

########################################################################################

def recoverSentence(ids, length, decoder_dict, tokenization):
    ids = ids[0].tolist()
    length = length[0].item()
    sentence = ids2words(ids, decoder_dict, tokenization)
    sentence = ' '.join(sentence[:length])
    return sentence

########################################################################################

def main(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = device
    tokenization = args.tokenization
    if args.mode in ["vis", "ins"]:
        args.batch_size = 1
    # load model parameters from checkpoint
    loaded = torch.load(args.ckpt, map_location={'cuda:0':'cpu'})  
    model_kwargs = loaded['model_kwargs']
    model_kwargs['mode'] = args.mode
    model_kwargs['molecule'] = 'ligand'
    for k, v in model_kwargs.items():
        setattr(args, k, v)
    task = model_kwargs['task']
    data = data_loaderX(args)   
    modelX = ARTM_model
    model = modelX(args)
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    # print(f'\n    |    # of Word Embed Params  : {num_embedding_params}     |')
    # print(f'    |    # of Params exclude W. Embed: {num_params - num_embedding_params}     |')
    # print(f'    |>>  # of TOTAL PARAMETERS   : {num_params}  <<|\n')    
    model.load_state_dict(loaded['model'])
    model.eval()
    model = model.to(device)
    if task == "clf":
        test_names = [""]   # actually it is path not just name.
        tag = "CE"
    elif task == "reg":
        test_names = [""]   # actually it is path not just name.
        tag = "MSE"
    #######################################################################################################################
    #######################################################################################################################   
    if args.mode == 'val':
        print('    |  Validating on test sets..  |\n')        
        for test_name in test_names:
            # test_path = str(args.df_loc) + str(test_name) + "_test.csv"
            test_path = str(args.df_loc) + "_test.csv"
            test_dataset = pd.read_csv(test_path)
            test_dataset.reset_index(inplace=True, drop=True)
            num_test_batches = math.ceil(len(test_dataset) / args.batch_size)
            test_size = len(test_dataset)
            print(f'\n    |  TEST SET NAME: "{test_name}"  |  TEST SET SIZE: {test_size} data  |\n')
            with tqdm(total=(num_test_batches), unit="batch") as pbar_test:
                total_correct = 0
                test_loss_list = []
                predictions = []
                ground_truth = []
                for test_batch in data.test_minibatch_generator(test_dataset, args.mode):
                    if args.task == "clf":
                        _, test_loss, curr_correct, labels, preds = eval_iter(test_batch, model, task, args.mode)
                        total_correct += curr_correct
                        predictions.extend(preds)
                        ground_truth.extend(labels)
                        test_loss_list.append(test_loss.item())
                        # pbar_test.set_description(f'    |  "{test_name}" Test {tag}Loss: {test_loss.item():.4f}  |  Loading..')
                    elif args.task == "reg":
                        _, test_loss = eval_iter(args, test_batch, *trpack)
                        test_loss_list.append(test_loss.item())
                roc_score = np.round(roc_auc_score(ground_truth, predictions), 4)
                precision, recall, _ = precision_recall_curve(ground_truth, predictions)
                prc_score = np.round(auc(recall, precision), 4)
                test_loss_mean = torch.mean(torch.Tensor(test_loss_list)).item()
                test_accuracy = (total_correct / len(test_dataset)) * 100
                # pbar_test.set_description(f'    |  "{test_name}" Test {tag}Loss: {test_loss_mean:.4f}  |  Accuracy: % {test_accuracy:.2f}')
        print(f"\n    |  TEST HAS BEEN COMPLETED!  |  Test ROC-AUC = % {roc_score:.4f}  |  Test PRC-AUC = % {prc_score:.4f}  |\n")
    #######################################################################################################################
    #######################################################################################################################
    elif args.mode == "vis":
        print('Visualizing learned tree structures..')
        print("\n")
        print('='*50)
        if tokenization == "bpe":
            with open("vocabs/chemical/chembl27_bpe_32000.json", "r") as f1:
                vis_decoder_chem_dict = json.load(f1)
            vis_decoder_chem_dict = invert_dict(vis_decoder_chem_dict["model"]["vocab"])
            with open("vocabs/protein/uniprot_bpe_32000.json", "r") as f2:
                vis_decoder_prot_dict = json.load(f2)
            vis_decoder_prot_dict = vis_decoder_chem_dict #   invert_dict(vis_decoder_prot_dict["model"]["vocab"])
        elif tokenization == "cha":
            with open("data/INV_CHARSET.json", "r") as f:
                inv_charset = json.load(f)
            vis_decoder_chem_dict = inv_charset["INV_SMISET"]
            vis_decoder_prot_dict = vis_decoder_chem_dict #   inv_charset["INV_PROTSET"]           
        counter = 0
        test_path = str(args.df_loc) + "_test.csv"
        test_dataset = pd.read_csv(test_path)
        test_dataset.reset_index(inplace=True, drop=True)
        freq_dict_ligand = {}
        for test_batch in data.test_minibatch_generator(test_dataset, args.mode):
            # model_arg, _, chem_id = test_batch[0]
            model_arg = test_batch[0]
            # print(test_batch[1])
            # chem_id = test_batch[3].tolist()[0]   # OLMAMALII..
            # print("Chem ID:", chem_id)   # OLMAMALII..
            chem_id = counter
            logits, supplements = model(**model_arg)
            visualizeTree(postOrder(supplements['tree'][0]))                
            print("\n", recoverSentence(model_arg['ligand'], model_arg['length'], vis_decoder_chem_dict, tokenization))
            freq_dict_ligand = gradeNodes(postOrder(supplements['tree'][0]), freq_dict_ligand, chem_id, test_loss)
            counter += 1
            print("\n", counter)
            print('='*50)
        with open("freq_dict_ligand.json", "w") as f:
            json.dump(freq_dict_ligand, f)
        print("\n    |  VISUALIZATION HAS BEEN COMPLETED!  |\n")
    #######################################################################################################################
    #######################################################################################################################
    elif args.mode == "ins":       
        if tokenization == "bpe":
            with open("vocabs/chemical/chembl27_bpe_32000.json", "r") as f1:
                vis_decoder_chem_dict = json.load(f1)
            vis_decoder_chem_dict = invert_dict(vis_decoder_chem_dict["model"]["vocab"])
            with open("vocabs/protein/uniprot_bpe_32000.json", "r") as f2:
                vis_decoder_prot_dict = json.load(f2)
            vis_decoder_prot_dict = vis_decoder_chem_dict #   invert_dict(vis_decoder_prot_dict["model"]["vocab"])
        elif tokenization == "cha":
            with open("data/INV_CHARSET.json", "r") as f:
                inv_charset = json.load(f)
            vis_decoder_chem_dict = inv_charset["INV_SMISET"]
            vis_decoder_prot_dict = vis_decoder_chem_dict #   inv_charset["INV_PROTSET"]
        with open("data/smi2cid.json", "r") as f:
            smi2cid = json.load(f)
        ############################################################## 
        counter = 0
        freq_dict_ligand = {}
        for i in range(3):   # kc in [122, 123, 124]:           
            test_path = str(args.df_loc) + "_test.csv"   # [:-3] + str(kc) + "_test.csv"
            print("\n>>", test_path[5:], "file is started for inspection.")
            test_dataset = pd.read_csv(test_path)
            test_dataset.reset_index(inplace=True, drop=True)
            test_size = len(test_dataset)
            ##############################################################            
            for test_batch in data.test_minibatch_generator(test_dataset, args.mode):
                model_arg = test_batch[0]
                chem_id = str(test_batch[2].iloc[0])   # aslında smiles!
                chem_id = smi2cid[chem_id]
                # print(chem_id) 
                _, supplements = model(**model_arg)
                if args.task == "clf":
                    _, test_loss, _, _, _ = eval_iter(test_batch, model, task, args.mode)
                    test_loss = test_loss.item()          
                elif args.task == "reg":
                    _, test_loss = eval_iter(args, test_batch, *trpack)
                    test_loss = test_loss.item()
                print('='*50)
                print(">> Ligand", chem_id, "\n")
                visualizeTree(postOrder(supplements['tree'][0]))               
                print("\n", recoverSentence(model_arg['ligand'], model_arg['length'], vis_decoder_chem_dict, tokenization))
                print('='*50, "\n")
                freq_dict_ligand = gradeNodes(postOrder(supplements['tree'][0]), freq_dict_ligand, chem_id, test_loss)
                counter += 1
                # print("\n", counter, "Test Loss =", np.round(test_loss, 4))
                # print('='*50)
        output_name = "freq_dicts/freq_dict_ligand_" + str(args.df_loc)[5:-3] + ".json"
        with open(output_name, "w") as f:
            json.dump(freq_dict_ligand, f)                    
        print("\n    |  INSPECTION HAS BEEN COMPLETED!  |\n")
    elif args.mode == "emb":
        test_path = args.df_loc + "_test.csv"   # "data/CB_uniques.csv"
        train_path = args.df_loc + "_train.csv"
        test_dataset = pd.read_csv(test_path)
        train_dataset = pd.read_csv(train_path)
        test_dataset = pd.concat([train_dataset, test_dataset], axis=0, ignore_index=True)
        num_test_batches = math.ceil(len(test_dataset) / args.batch_size)        
        embeddings = []
        chem_ids_list = []
        with tqdm(total=(num_test_batches), unit="batch") as pbar_emb:
            for test_batch in data.test_minibatch_generator(test_dataset, args.mode):
                chem_ids = test_batch[2]   # .tolist()
                logits = eval_iter(test_batch, model, task, args.mode)
                pbar_emb.update(1)
                pbar_emb.set_description('    |  Creating embeddings..')
                embeddings.extend(logits.cpu().detach().numpy().tolist())
                # chem_ids_list.extend(chem_ids) #chem_ids_list.extend(chem_ids.cpu().detach().numpy().tolist())
        df = pd.DataFrame({"embedding": embeddings})   # "ligand_id": chem_ids_list, 
        emb_name = "e_out/" + args.file_name + "_emb.pkl"
        file = open(emb_name, 'ab')
        pickle.dump(df, file)
        file.close()
        print("\n    |  EMBEDDINGS HAVE BEEN CREATED!  |\n")
    #######################################################################################################################
    #######################################################################################################################      

########################################################################################

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default="")
    parser.add_argument('--ckpt', default="")
    # parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--tokenization', default='bpe', choices=['bpe', 'cha'])
    parser.add_argument('--mode', default='val', choices=['vis', 'val', 'ins'])   # , 'emb'
    parser.add_argument('--batch-size', default=32, type=int)   
    parser.add_argument('--task', default='clf', choices=['clf'])   # , 'reg'
    parser.add_argument('--file_name', default="")
    args = parser.parse_args() 
    return args

########################################################################################

if __name__ == '__main__':
    ########################################################################################
    args = load_args()
    ########################################################################################
    for file_orj in os.listdir("results"):   # THIS IS AN ALL_in_ONE PROCEDURE !
        if file_orj[-4:] != ".pkl":
            continue       
        args.ckpt = "results/" + file_orj
        args.file_name = file_orj.split("-")[0]
        args.df_loc = "data/" + args.file_name   # + "_test.csv"
        print("\n>>", args.file_name, "is started to embedding creating procedure.\n")
        main(args)
    ########################################################################################
        
########################################################################################
