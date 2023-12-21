import torch
import numpy as np
import sys
import re
import os
import json
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class data_loader(object):
    def __init__(self, args):
        self.args = args
        self.train_loc = f"{args.data_folder}/{args.data_name}/{args.data_name}_train.csv"
        self.val_loc = f"{args.data_folder}/{args.data_name}/{args.data_name}_val.csv"
        self.test_loc = f"{args.data_folder}/{args.data_name}/{args.data_name}_test.csv"
        self.all_loc = f"{args.data_folder}/{args.data_name}/{args.data_name}_all.csv"
        self.train_dataset = pd.read_csv(self.train_loc)       
        self.train_dataset.reset_index(inplace=True, drop=True)  
        self.valid_dataset = pd.read_csv(self.val_loc)       
        self.valid_dataset.reset_index(inplace=True, drop=True)  
        self.test_dataset = pd.read_csv(self.test_loc)       
        self.test_dataset.reset_index(inplace=True, drop=True)  
        self.all_dataset = pd.read_csv(self.all_loc)       
        self.all_dataset.reset_index(inplace=True, drop=True)
        self.label_name = args.y_label
        self.input_name = args.x_label
        with open(f"{args.data_folder}/CHARSET.json", "r") as f: self.word_to_id_l = json.load(f)
        self.train_label = self.train_dataset[self.label_name].tolist()
        self.valid_label = self.valid_dataset[self.label_name].tolist()  
        self.test_label = self.test_dataset[self.label_name].tolist()
        self.all_label = self.all_dataset[self.label_name].tolist()
        self.train_chem = self.encode_cha_dataset(self.train_dataset[self.input_name].tolist())
        self.train_dataset = [(self.train_chem[i], self.train_label[i], self.train_dataset[self.input_name][i]) for i in range(len(self.train_dataset))]                 
        self.valid_chem = self.encode_cha_dataset(self.valid_dataset[self.input_name].tolist())    
        self.valid_dataset = [(self.valid_chem[i], self.valid_label[i], self.valid_dataset[self.input_name][i]) for i in range(len(self.valid_dataset))]
        self.test_chem = self.encode_cha_dataset(self.test_dataset[self.input_name].tolist())    
        self.test_dataset = [(self.test_chem[i], self.test_label[i], self.test_dataset[self.input_name][i]) for i in range(len(self.test_dataset))]
        self.all_chem = self.encode_cha_dataset(self.all_dataset[self.input_name].tolist())    
        self.all_dataset = [(self.all_chem[i], self.all_label[i], self.all_dataset[self.input_name][i]) for i in range(len(self.all_dataset))]
        self.id_to_word_l = {v: k for k, v in self.word_to_id_l.items()}       
        self.num_train_batches = math.ceil(len(self.train_dataset) / args.batch_size)
        self.num_valid_batches = math.ceil(len(self.valid_dataset) / args.batch_size) 
        self.num_test_batches = math.ceil(len(self.test_dataset) / args.batch_size)
        self.num_all_batches = math.ceil(len(self.all_dataset) / args.batch_size)
        self.train_size = len(self.train_dataset)
        self.valid_size = len(self.valid_dataset)   
        self.test_size = len(self.test_dataset)
        self.all_size = len(self.all_dataset)
        print(f"\n\n    |  TRAIN SET SIZE: {self.train_size} data  |")
        print(f"    |  VALID SET SIZE: {self.valid_size} data  |")      
        print(f"    |  TEST SET SIZE: {self.test_size} data  |\n\n")
        self.vocab_l = self.word_to_id_l.keys()                
        self.weight = None 
        args.num_words = len(self.vocab_l)
        args.vocab = self
    
    def wrap_to_model_arg(self, chemicals, l_chems):
        return {"encoded_ligand": torch.LongTensor(chemicals).to(self.args.device),
                "ligand_length": torch.LongTensor(l_chems).to(self.args.device)}
    
    def smiles_segmenter(self, smi):
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        smi = smi.replace(" ", "")
        tokens = [token for token in regex.findall(smi)]
        smi_new = "".join(tokens)    
        if smi != smi_new: print(f"\n{smi}\n{smi_new}")
        assert smi == smi_new
        return tokens

    def encode_cha_smiles(self, seq):
        seq = self.smiles_segmenter(seq)
        DATA_LEN = len(seq)
        if DATA_LEN > self.args.max_smi_len: DATA_LEN = self.args.max_smi_len
        labeled_data = np.zeros(DATA_LEN, dtype=int)
        for i, cha in enumerate(seq[:DATA_LEN]): labeled_data[i] = self.word_to_id_l[cha]
        return labeled_data
    
    def encode_cha_dataset(self, data_list):
        encoded_list = []
        for data in data_list: encoded_list.append(self.encode_cha_smiles(data.strip())) 
        return np.array(encoded_list)
    
    def get_cha_data(self, data):
        data = data.copy()
        data = pd.DataFrame(data)
        batch_size = data.shape[0]
        chemicals = np.zeros((batch_size, self.args.max_smi_len), dtype="int32")  
        l_chems = np.zeros((batch_size,), dtype="int32")
        labels = np.zeros((batch_size,), "float32")          
        for i in range(batch_size):
            chemicals[i, :len(data[0][i])] = data[0][i]          
            l_chems[i] = len(data[0][i])
            labels[i] = data[1][i]
        smis = data[2]
        return np.array(chemicals), np.array(labels), np.array(l_chems), smis      

    def generator(self, process):
        process_conv = {"train": self.train_dataset, "valid": self.valid_dataset, "test": self.test_dataset, "all": self.all_dataset}
        data = process_conv[process]
        data = pd.DataFrame(data)
        data = data.sample(frac = 1)
        ptr = 0
        len_data = len(data)
        while ptr < len_data:
            batch_size = min(self.args.batch_size, (len_data - ptr))
            ptr += self.args.batch_size
            minibatch = data[(ptr - self.args.batch_size) : ptr]
            minibatch.reset_index(inplace=True, drop=True) 
            chemicals, labels, l_chems, smis = self.get_cha_data(minibatch)
            if self.args.task == "clf": labels = torch.LongTensor(labels).to(self.args.device)
            elif self.args.task == "reg": labels = torch.FloatTensor(labels).to(self.args.device)
            model_arg = self.wrap_to_model_arg(chemicals, l_chems)
            yield model_arg, labels, smis
      