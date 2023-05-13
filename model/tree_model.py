########################################################################################
########################################################################################

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from model.STGumbel_AR_Tree import STG_AR_Tree
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, roc_auc_score   # , confusion_matrix,
import numpy as np

########################################################################################
########################################################################################

def calc_metrics(y_true, y_pred, y_prob, data):
    ##########################
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    ##########################
    # tpr = tp / (tp + fn)   # = sensitivity = recall
    # fpr = fp / (tn + fp)   # 1-specifity
    # precision = tp / (tp + fp)
    ##########################
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ##########################
    roc_score = np.round(auc(fpr, tpr), 4)
    prc_score = np.round(auc(recall, precision), 4)
    accuracy = np.round(accuracy_score(y_true, y_pred), 4)
    ##########################
    return roc_score, prc_score, accuracy

########################################################################################
########################################################################################

class Classifier(nn.Module):
    ########################################################################################
    
    def __init__(self, args):
        super().__init__()
        ##########################
        self.args = args
        ##########################
        if self.args.act_func == "ReLU":
            self.actv_func = nn.ReLU()
        elif self.args.act_func == "Tanh":
            self.actv_func = nn.Tanh()
        ##########################
        if args.use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=args.tree_hidden_dim)   # ?
            self.bn_mlp_output = nn.BatchNorm1d(num_features=args.DTA_hidden_dim)   # ?
        ##########################
        self.dense1 = nn.Linear(args.tree_hidden_dim, args.DTA_hidden_dim)
        ##########################
        self.fcs = []
        for i in range(args.clf_num_layers):
            ##########################
            if i == (args.clf_num_layers - 1):
                self.fcs.append(nn.Linear(args.DTA_hidden_dim, (args.DTA_hidden_dim // 2)))
                self.fcs.append(self.actv_func)
                # self.fcs.append(nn.Dropout(args.dropout))
                break
            ##########################
            self.fcs.append(nn.Linear(args.DTA_hidden_dim, args.DTA_hidden_dim))
            self.fcs.append(self.actv_func)
            self.fcs.append(nn.Dropout(args.dropout))
        ##########################
        self.fcs = nn.Sequential(*self.fcs)
        ##########################  
        self.out = nn.Linear((args.DTA_hidden_dim // 2), 1)
        ##########################
        ##########################
        self.reset_parameters()
        ##########################
    
    ########################################################################################
    
    def reset_parameters(self):
        ##########################
        if self.args.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        ##########################
        for m in self.fcs:
            ##########################   STG + BRK
            if type(m) == nn.Linear:
                init.kaiming_normal_(m.weight.data)
                init.constant_(m.bias, val=0)
            ##########################   NRL
            # if isinstance(m, nn.Linear):
                # init.xavier_uniform_(m.weight)
                # init.normal_(m.weight.data, mean=0, std=0.01) ??? for word_emb
                # m.bias.data.fill_(0.01)
        ##########################    STG + BRK
        init.kaiming_normal_(self.out.weight.data)
        init.constant_(self.out.bias, val=0)
        ##########################   NRL
        # init.xavier_uniform_(m.weight)
        # init.normal_(m.weight.data, mean=0, std=0.01) ??? for word_emb
        # m.bias.data.fill_(0.01)
        ##########################
    
    ########################################################################################
    
    def forward(self, sentence):
        x = self.dense1(sentence)
        x = self.fcs(x)
        x = self.out(x)   # if it is clf, than x = probabilities, not logits. because there will be sigmoid func after the final layer
        return x
        
    ########################################################################################
    
######################################################################################## 
########################################################################################   

class ARTM_model(nn.Module):
    ########################################################################################
    
    def __init__(self, args):
        super().__init__()
        ##########################
        self.args = args
        ##########################
        self.dropout = nn.Dropout(args.dropout)
        Encoder = STG_AR_Tree
        self.word_embedding = nn.Embedding(num_embeddings=args.num_words, embedding_dim=args.word_dim)                                 
        self.encoder = Encoder(args)
        self.classifier = Classifier(args)
        if args.task == "clf":
            self.sigmoid = nn.Sigmoid()
        ##########################
        self.reset_parameters()
        ##########################
    
    ########################################################################################
    
    def reset_parameters(self):
        init.normal_(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()
    
    ########################################################################################
    
    def forward(self, encoded_ligand, ligand_length):   # batch bunlar !
        words_embed = self.word_embedding(encoded_ligand)
        words_embed = self.dropout(words_embed)
        h, _, tree = self.encoder(words_embed, encoded_ligand, ligand_length)
        if self.args.mode == "emb":
            return h, []
        supplements = {"tree": tree}
        x = self.classifier(h)
        if self.args.task == "clf":
            x = self.sigmoid(x)
        elif self.args.task == "reg":
            pass   # nothing is needed.
        return x, supplements
        
    ########################################################################################
    
########################################################################################
########################################################################################














