import torch
import numpy as np
import sys
import re
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizers import Tokenizer
import json
import pandas as pd
# pd.options.mode.chained_assignment = None
import math
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class TheData(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.device = args.device
        self.tokenization = args.tokenization
        self.task = args.task
        self.max_prot_len = 1000
        self.max_smi_len = 100  
        self.df_loc = args.df_loc
        self.train_loc = str(self.df_loc) + "_train.csv"
        self.val_loc =  str(self.df_loc) + "_val.csv"       
        self.train_dataset = pd.read_csv(self.train_loc)       
        self.train_dataset.reset_index(inplace=True, drop=True)  
        self.valid_dataset = pd.read_csv(self.val_loc)       
        self.valid_dataset.reset_index(inplace=True, drop=True)              
        if self.tokenization == "cha":
            with open("data/CHARSET.json", "r") as f1:
                self.CHARSET = json.load(f1)
            with open("data/INV_CHARSET.json", "r") as f2:
                self.inv_charset = json.load(f2)
            self.CHARPROTSET = self.CHARSET["CHARPROTSET"]
            self.CHARSMISET = self.CHARSET["CHARSMISET"]
            self.inv_charsmiset = self.inv_charset["INV_SMISET"]
            self.inv_charprotset = self.inv_charset["INV_PROTSET"] 
            self.train_label = self.train_dataset['affinity_score'].tolist()
            self.valid_label = self.valid_dataset['affinity_score'].tolist()  
            self.train_chem = self.encode_data(self.train_dataset['smiles'].tolist(), self.max_smi_len, self.CHARSMISET)
            temp_train = []
            for i in range(len(self.train_dataset)):
                temp_train.append((self.train_chem[i], self.train_label[i]))
            self.train_dataset = temp_train                   
            self.valid_chem = self.encode_data(self.valid_dataset['smiles'].tolist(), self.max_smi_len, self.CHARSMISET)    
            temp_valid = []
            for i in range(len(self.valid_dataset)):
                temp_valid.append((self.valid_chem[i], self.valid_label[i]))
            self.valid_dataset = temp_valid            
            self.word_to_id_p = self.CHARPROTSET                 
            self.id_to_word_p = self.inv_charprotset #{v: k for k, v in self.word_to_id_p.items()}
            self.word_to_id_l = self.CHARSMISET
            self.id_to_word_l = self.inv_charsmiset #{v: k for k, v in self.word_to_id_l.items()}
        ########################################################################################
        elif self.tokenization == "bpe":
            from word_identifier import WordIdentifier        
            self.encoding_vocab_path = "vocabs/chemical/chembl27.vocab"       
            self.chem_tokenizer = WordIdentifier.from_file("vocabs/chemical/chembl27_enc_bpe_32000.json")
            self.prot_tokenizer = WordIdentifier.from_file("vocabs/protein/uniprot_bpe_32000.json")           
            with open("vocabs/chemical/chembl27_bpe_32000.json", "r") as f1:
                l_data = json.load(f1)
            with open("vocabs/protein/uniprot_bpe_32000.json", "r") as f2:    
                p_data = json.load(f2)                    
            self.word_to_id_p = p_data["model"]["vocab"] 
            self.id_to_word_p = {v: k for k, v in self.word_to_id_p.items()}
            self.word_to_id_l = l_data["model"]["vocab"]
            self.id_to_word_l = {v: k for k, v in self.word_to_id_l.items()}
        self.vocab_l = self.word_to_id_l.keys()                
        self.weight = None                
        self.num_train_batches = math.ceil(len(self.train_dataset) / self.batch_size)
        self.num_valid_batches = math.ceil(len(self.valid_dataset) / self.batch_size)        
        self.train_size = len(self.train_dataset)
        self.valid_size = len(self.valid_dataset)        
        print(f"\n    |  TRAIN SET SIZE: {self.train_size} data  |")
        print(f"    |  VALID SET SIZE: {self.valid_size} data  |\n")        
        args.num_words = 32000
        args.vocab = self             
    
    def wrap_numpy_to_longtensor(self, *args):
        res = []
        for arg in args:
            arg = torch.LongTensor(arg).to(self.device)
            res.append(arg)
        return res

    def wrap_to_model_arg(self, chemicals, proteins, l_chems, l_prots): # should match the kwargs of model.forward
        return {'ligand': chemicals,
                'length':l_chems}
    
    def encode_data(self, data_list, MAX_LEN, char_set):
        encoded_list = []
        for data in data_list:
            encoded_list.append(self.label_data(data.strip(), MAX_LEN, char_set))  
        return np.array(encoded_list)
    
    def label_data(self, seq, MAX_LEN, char_set):
        if char_set == self.CHARSMISET:
            seq = self.smiles_segmenter(seq)
        DATA_LEN = len(seq)
        if DATA_LEN > MAX_LEN:
            DATA_LEN = MAX_LEN
        labeled_data = np.zeros(DATA_LEN, dtype=int)
        for i, cha in enumerate(seq[:DATA_LEN]):
            labeled_data[i] = char_set[cha]
        return labeled_data
        
    def smiles_segmenter(self, smi):
        pattern = '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        if smi != ''.join(tokens):
            print(smi)
        assert smi == ''.join(tokens)
        return tokens

    def encode_bpe_smiles(self, smiles, encoding_vocab_path):
        segments = self.smiles_segmenter(smiles)
        with open(encoding_vocab_path) as f:
            encoding_vocab = json.load(f)
        output = ''.join([encoding_vocab.get(segment, encoding_vocab['[OOV]']) for segment in segments])
        return output

    def get_bpe_data(self, data):
        data = data.copy()        
        data_smiles = data['smiles'].apply(self.encode_bpe_smiles, encoding_vocab_path=self.encoding_vocab_path)
        chemicals, l_chems = self.chem_tokenizer.identify_words(data_smiles, padding_len=self.max_smi_len, out_type='int', seq_type="smi")       
        proteins, l_prots = 1, 2 #self.prot_tokenizer.identify_words(data["aa_sequence"], padding_len=self.max_prot_len, out_type='int', seq_type="prot")
        labels = 1   # data["affinity_score"]
        return np.array(chemicals), np.array(proteins), np.array(labels), np.array(l_chems), np.array(l_prots)
        
    def get_cha_data(self, data, mode=""):
        data = data.copy()
        data = pd.DataFrame(data)
        batch_size = data.shape[0]
        chemicals = np.zeros((batch_size, self.max_smi_len), dtype='int32')
        proteins = np.zeros((batch_size, self.max_prot_len), dtype='int32')     
        l_chems = np.zeros((batch_size,), dtype='int32')
        l_prots = np.zeros((batch_size,), dtype='int32')
        labels = np.zeros((batch_size,), 'float32')          
        for i in range(batch_size):
            chemicals[i, :len(data[0][i])] = data[0][i]          
            l_chems[i] = len(data[0][i])
            labels[i] = data[1][i]               
        return np.array(chemicals), np.array(proteins), np.array(labels), np.array(l_chems), np.array(l_prots)        

    def train_minibatch_generator(self):
        self.train_ptr = 0
        self.train_dataset = pd.DataFrame(self.train_dataset)
        self.train_dataset = self.train_dataset.sample(frac=1) #for randomization
        while self.train_ptr < self.train_size:
            batch_size = min(self.batch_size, self.train_size - self.train_ptr) 
            self.train_ptr += batch_size
            minibatch = self.train_dataset[self.train_ptr - batch_size : self.train_ptr]
            if self.tokenization == "bpe":
                chemicals, proteins, labels, l_chems, l_prots = self.get_bpe_data(minibatch)
            elif self.tokenization == "cha":
                minibatch.reset_index(inplace=True, drop=True) 
                chemicals, proteins, labels, l_chems, l_prots = self.get_cha_data(minibatch)
            chemicals, proteins, l_chems, l_prots = self.wrap_numpy_to_longtensor(chemicals, proteins, l_chems, l_prots)
            labels = torch.LongTensor(labels).to(self.device)
            model_arg = self.wrap_to_model_arg(chemicals, proteins, l_chems, l_prots)
            yield model_arg, labels

    def valid_minibatch_generator(self):
        self.valid_ptr = 0
        self.valid_dataset = pd.DataFrame(self.valid_dataset)
        self.valid_dataset = self.valid_dataset.sample(frac=1) #for randomization
        while self.valid_ptr < self.valid_size:
            batch_size = min(self.batch_size, self.valid_size - self.valid_ptr)
            self.valid_ptr += batch_size
            minibatch = self.valid_dataset[self.valid_ptr - batch_size : self.valid_ptr]
            if self.tokenization == "bpe":
                chemicals, proteins, labels, l_chems, l_prots = self.get_bpe_data(minibatch)
            elif self.tokenization == "cha":
                minibatch.reset_index(inplace=True, drop=True) 
                chemicals, proteins, labels, l_chems, l_prots = self.get_cha_data(minibatch)
            chemicals, proteins, l_chems, l_prots = self.wrap_numpy_to_longtensor(chemicals, proteins, l_chems, l_prots)
            labels = torch.LongTensor(labels).to(self.device)
            model_arg = self.wrap_to_model_arg(chemicals, proteins, l_chems, l_prots)
            yield model_arg, labels
            
    def test_minibatch_generator(self, test_dataset, mode):           
        self.test_ptr = 0
        test_dataset = pd.DataFrame(test_dataset)
        self.test_size = len(test_dataset)         
        while self.test_ptr < self.test_size:
            batch_size = min(self.batch_size, self.test_size - self.test_ptr)
            self.test_ptr += batch_size
            minibatch = test_dataset[self.test_ptr - batch_size : self.test_ptr]
            if mode == "emb":
                chem_ids = []   # minibatch["ligand_id"]
            else:
                chem_ids = minibatch["smiles"]
            if self.tokenization == "bpe":
                chemicals, proteins, labels, l_chems, l_prots = self.get_bpe_data(minibatch)
            elif self.tokenization == "cha":
                minibatch.reset_index(inplace=True, drop=True)        
                cha_label = minibatch['affinity_score']
                cha_chem = self.encode_data(minibatch['smiles'].tolist(), self.max_smi_len, self.CHARSMISET)
                cha_prot = np.zeros((batch_size, self.max_prot_len), dtype='int32')       
                temp_cha = []
                for i in range(len(minibatch)):
                    temp_cha.append((cha_chem[i], cha_prot[i], cha_label[i]))
                minibatch = temp_cha
                chemicals, proteins, labels, l_chems, l_prots = self.get_cha_data(minibatch, "emb")  
            chemicals, proteins, l_chems, l_prots = self.wrap_numpy_to_longtensor(chemicals, proteins, l_chems, l_prots)
            labels = torch.LongTensor(labels).to(self.device)
            model_arg = self.wrap_to_model_arg(chemicals, proteins, l_chems, l_prots)
            yield model_arg, labels, chem_ids
        
