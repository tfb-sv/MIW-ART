from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
import json

with open("utils/vocabs/decoder_bpe_main_dict.json", "r") as f:
    decoder_dict_main = json.load(f)

class WordIdentifier:
    @classmethod
    def from_file(cls, path):
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        return instance
    
    def train_word_identifier_model(self, model_type, corpus_path, save_name, vocab_size, starting_vocab):
        print(f'WordIdentifier: Training a model with vocab {vocab_size} on {corpus_path}')
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['[PAD]'] + starting_vocab)
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.train([corpus_path], trainer)
        tokenizer.save(save_name, pretty=True)
        self.tokenizer = tokenizer

    def identify_words(self, sequences, padding_len=None, out_type='int', seq_type=None):    
        encodings = self.tokenizer.encode_batch(sequences)
        l_list = []
        # print("\n")
        if padding_len is not None:
            for encoding in encodings:
                # print("encoding.ids", encoding.ids)
                # print("encoding.ids.len", len(encoding.ids))
                # print("encoding.tokens", encoding.tokens)
                # print("encoding.tokens.len", len(encoding.tokens))
                # if seq_type == "prot":
                    # print("bped_prot", encoding.tokens)
                    # pass
                # if seq_type == "smi":
                    # print(encoding.tokens)
                    # print("bped_chem", encoding.tokens)
                    ######################################################################################
                    # new_token_seq = []
                    # for token in encoding.tokens:
                        # new_token = ""
                        # for symbol in token:
                            # letter = decoder_dict_main[symbol]
                            # new_token += letter
                        # new_token_seq.append(new_token)                   
                    # print("bped_chem", new_token_seq)
                    #######################################################################################
                if len(encoding) < padding_len:
                    l = len(encoding)
                else:
                    l = padding_len
                l_list.append(l)
                encoding.pad(padding_len, direction='right', pad_id=0, pad_token='[PAD]')
                encoding.truncate(padding_len)
                
        if out_type == 'int':
            return [encoding.ids for encoding in encodings], l_list
        elif out_type == 'str':
            return [encoding.tokens for encoding in encodings], l_list
        else:
            raise ValueError('Invalid out_type for word identification')
            