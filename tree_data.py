########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

import torch
import numpy as np
import sys
import re
import os
from tokenizers import Tokenizer
import json
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

########################################################################################
########################################################################################
########################################################################################

class data_loaderX(object):
    ########################################################################################
    ########################################################################################
    ########################################################################################
    
    def __init__(self, args):
        ########################################################################################
        self.args = args
        ##########################
        self.train_loc = (f"data/{args.data_name}/{args.data_name}_train.csv")
        self.val_loc = (f"data/{args.data_name}/{args.data_name}_val.csv")
        self.test_loc = (f"data/{args.data_name}/{args.data_name}_test.csv")
        self.all_loc = (f"data/{args.data_name}/{args.data_name}_all.csv")
        ##########################
        self.train_dataset = pd.read_csv(self.train_loc)       
        self.train_dataset.reset_index(inplace=True, drop=True)  
        ##########################
        self.valid_dataset = pd.read_csv(self.val_loc)       
        self.valid_dataset.reset_index(inplace=True, drop=True)  
        ##########################
        self.test_dataset = pd.read_csv(self.test_loc)       
        self.test_dataset.reset_index(inplace=True, drop=True)  
        ##########################
        try:
            self.all_dataset = pd.read_csv(self.all_loc)       
            self.all_dataset.reset_index(inplace=True, drop=True)
        except:
            self.all_dataset = pd.concat([self.train_dataset, self.valid_dataset, self.test_dataset], ignore_index=True, axis=0)
            self.all_dataset.reset_index(inplace=True, drop=True)
            self.all_dataset.to_csv(self.all_loc)
        ########################################################################################
        if args.tokenization == "cha":
            ##########################
            with open("data/CHARSET.json", "r") as f:
                self.CHARSMISET = json.load(f)
            with open("data/INV_CHARSET.json", "r") as f:
                self.inv_charsmiset = json.load(f)
            ##########################
            self.train_label = self.train_dataset["affinity_score"].tolist()
            self.valid_label = self.valid_dataset["affinity_score"].tolist()  
            self.test_label = self.test_dataset["affinity_score"].tolist()
            ##########################
            self.train_chem = self.encode_cha_dataset(self.train_dataset["smiles"].tolist())
            self.train_dataset = [(self.train_chem[i], self.train_label[i], self.train_dataset["smiles"][i]) for i in range(len(self.train_dataset))]                 
            ##########################
            self.valid_chem = self.encode_cha_dataset(self.valid_dataset["smiles"].tolist())    
            self.valid_dataset = [(self.valid_chem[i], self.valid_label[i], self.valid_dataset["smiles"][i]) for i in range(len(self.valid_dataset))]
            ##########################
            self.test_chem = self.encode_cha_dataset(self.test_dataset["smiles"].tolist())    
            self.test_dataset = [(self.test_chem[i], self.test_label[i], self.test_dataset["smiles"][i]) for i in range(len(self.test_dataset))]
            ##########################
            self.word_to_id_l = self.CHARSMISET
            self.id_to_word_l = self.inv_charsmiset
        ########################################################################################
        elif args.tokenization == "bpe":
            from utils.word_identifier import WordIdentifier  
            ##########################
            self.encoding_vocab_path = "utils/vocabs/chemical/chembl27.vocab"       
            self.chem_tokenizer = WordIdentifier.from_file("utils/vocabs/chemical/chembl27_enc_bpe_32000.json")   
            ##########################
            with open("utils/vocabs/chemical/chembl27_bpe_32000.json", "r") as f:
                l_data = json.load(f) 
            ##########################
            self.word_to_id_l = l_data["model"]["vocab"]
            self.id_to_word_l = {v: k for k, v in self.word_to_id_l.items()}
        ######################################################################################## ???
        self.vocab_l = self.word_to_id_l.keys()                
        self.weight = None   # ???           
        self.num_train_batches = math.ceil(len(self.train_dataset) / args.batch_size)
        self.num_valid_batches = math.ceil(len(self.valid_dataset) / args.batch_size) 
        self.num_test_batches = math.ceil(len(self.test_dataset) / args.batch_size)
        self.train_size = len(self.train_dataset)
        self.valid_size = len(self.valid_dataset)   
        self.test_size = len(self.test_dataset)
        print(f"\n\n    |  TRAIN SET SIZE: {self.train_size} data  |")
        print(f"    |  VALID SET SIZE: {self.valid_size} data  |")      
        print(f"    |  TEST SET SIZE: {self.test_size} data  |\n\n")
        args.num_words = len(self.vocab_l)
        args.vocab = self   # ???
        ######################################################################################## ???
        
    ########################################################################################
    ########################################################################################
    ########################################################################################
    
    def wrap_to_model_arg(self, chemicals, l_chems):   # for both cha and bpe
        return {"encoded_ligand": torch.LongTensor(chemicals).to(self.args.device),
                "ligand_length": torch.LongTensor(l_chems).to(self.args.device)}
    
    ########################################################################################
    
    def smiles_segmenter(self, smi):   # for both cha and bpe
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        if smi != "".join(tokens):
            print(smi)
        assert smi == "".join(tokens)
        # if self.args.tokenization == "cha":
        tokens2 = []
        for token in tokens:
            token2 = token.replace("(", "{")
            token2 = token2.replace(")", "}")
            tokens2.append(token2)
        tokens = tokens2
        return tokens
    
    ########################################################################################

    def encode_cha_smiles(self, seq):   # for only cha (kırpmak için, bpe"de ihtiyaç yok...)
        seq = self.smiles_segmenter(seq)
        DATA_LEN = len(seq)
        if DATA_LEN > self.args.max_smi_len:
            DATA_LEN = self.args.max_smi_len
        labeled_data = np.zeros(DATA_LEN, dtype=int)
        for i, cha in enumerate(seq[:DATA_LEN]):
            labeled_data[i] = self.CHARSMISET[cha]
        return labeled_data
    
    ########################################################################################
    
    def encode_bpe_smiles(self, smiles, encoding_vocab_path):   # for only bpe
        segments = self.smiles_segmenter(smiles)
        with open(encoding_vocab_path) as f:
            encoding_vocab = json.load(f)
        output = "".join([encoding_vocab.get(segment, encoding_vocab["[OOV]"]) for segment in segments])
        return output
    
    ########################################################################################
    
    def encode_cha_dataset(self, data_list):   # for only cha
        encoded_list = []
        for data in data_list:
            encoded_list.append(self.encode_cha_smiles(data.strip())) 
        return np.array(encoded_list)
    
    ########################################################################################
    
    def get_bpe_data(self, data):   # for only bpe
        data = data.copy()        
        data_smiles = data["smiles"].apply(self.encode_bpe_smiles, encoding_vocab_path=self.encoding_vocab_path)
        chemicals, l_chems = self.chem_tokenizer.identify_words(data_smiles, padding_len=self.args.max_smi_len, out_type="int", seq_type="smi")       
        labels = data["affinity_score"]
        smis = data["smiles"]
        return np.array(chemicals), np.array(labels), np.array(l_chems), smis
    
    ########################################################################################
    
    def get_cha_data(self, data):   # for only cha (sadece batching yapıyor..)
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
    
    ########################################################################################
    ########################################################################################
    ########################################################################################
    
    def generator(self, process):
        ########################################################################################
        process_conv = {"train": self.train_dataset, "valid": self.valid_dataset, "test": self.test_dataset, "all": self.all_dataset}
        ##########################
        data = process_conv[process]
        ##########################
        data = pd.DataFrame(data)
        data = data.sample(frac = 1)   # for randomization
        ##########################
        ptr = 0
        len_data = len(data)
        ##########################
        while ptr < len_data:
            ##########################
            batch_size = min(self.args.batch_size, (len_data - ptr))
            ptr += self.args.batch_size
            ##########################
            minibatch = data[(ptr - self.args.batch_size) : ptr]
            ##########################
            if self.args.tokenization == "bpe":
                chemicals, labels, l_chems, smis = self.get_bpe_data(minibatch)
            elif self.args.tokenization == "cha":
                minibatch.reset_index(inplace=True, drop=True) 
                chemicals, labels, l_chems, smis = self.get_cha_data(minibatch)
            ##########################
            labels = torch.LongTensor(labels).to(self.args.device)
            model_arg = self.wrap_to_model_arg(chemicals, l_chems)
            ##########################
            yield model_arg, labels, smis
        ########################################################################################
    
    ########################################################################################
    ########################################################################################
    ########################################################################################
    
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################




















        
