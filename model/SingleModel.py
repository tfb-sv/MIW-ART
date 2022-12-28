import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.use_batchnorm = kwargs['use_batchnorm']
        self.task = kwargs['task']
        input_dim = kwargs['tree_hidden_dim'] # it should be the szie of ligand/protein after AR tree
        hidden_dim = kwargs['DTA_hidden_dim']
        self.dropout_rate = kwargs['dropout']
        if self.use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense1 = nn.Linear(in_features= input_dim, out_features=hidden_dim)
        self.dense2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dense3 = nn.Linear(in_features=hidden_dim, out_features=(hidden_dim//2))
        if self.task == "clf":
            self.out_layer = nn.Linear(in_features=(hidden_dim//2), out_features=2)   # etrafından dolaşıyoruz, önemli değil, sıkıntı çıkabilir diye duruyor
            # self.new_out = nn.Linear(in_features=(hidden_dim//2), out_features=2)
        elif self.task == "reg":        
            self.out_layer = nn.Linear(in_features=(hidden_dim//2), out_features=1)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        init.kaiming_normal_(self.dense1.weight) 
        init.constant_(self.dense1.bias, val=0)
        init.kaiming_normal_(self.dense2.weight)
        init.constant_(self.dense2.bias, val=0)
        init.kaiming_normal_(self.dense3.weight)
        init.constant_(self.dense3.bias, val=0)
        init.kaiming_normal_(self.out_layer.weight)
        init.constant_(self.out_layer.bias, val=0)
        # init.kaiming_normal_(self.new_out.weight)
        # init.constant_(self.new_out.bias, val=0)

    def forward(self, sentence):
        out = F.dropout(torch.relu(self.dense1(sentence)), self.dropout_rate)
        out = F.dropout(torch.relu(self.dense2(out)), self.dropout_rate)
        out = torch.relu(self.dense3(out))
        out = self.out_layer(out)
        # out = self.new_out(out)     
        return out

class New_Classifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.use_batchnorm = kwargs['use_batchnorm']
        self.task = kwargs['task']
        input_dim = kwargs['tree_hidden_dim'] # it should be the szie of ligand/protein after AR tree
        hidden_dim = kwargs['DTA_hidden_dim']
        self.dropout_rate = kwargs['dropout']
        if self.use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense1 = nn.Linear(in_features= input_dim, out_features=hidden_dim)
        self.dense2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dense3 = nn.Linear(in_features=hidden_dim, out_features=(hidden_dim//2))
        if self.task == "clf":
            self.out_layer = nn.Linear(in_features=(hidden_dim//2), out_features=2)   # etrafından dolaşıyoruz, önemli değil, sıkıntı çıkabilir diye duruyor
            # self.new_out = nn.Linear(in_features=(hidden_dim//2), out_features=2)
        elif self.task == "reg":        
            self.out_layer = nn.Linear(in_features=(hidden_dim//2), out_features=1)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        init.kaiming_normal_(self.dense1.weight) 
        init.constant_(self.dense1.bias, val=0)
        init.kaiming_normal_(self.dense2.weight)
        init.constant_(self.dense2.bias, val=0)
        init.kaiming_normal_(self.dense3.weight)
        init.constant_(self.dense3.bias, val=0)
        init.kaiming_normal_(self.out_layer.weight)
        init.constant_(self.out_layer.bias, val=0)
        # init.kaiming_normal_(self.new_out.weight)
        # init.constant_(self.new_out.bias, val=0)

    def forward(self, sentence):
        out = F.dropout(torch.relu(self.dense1(sentence)), self.dropout_rate)
        out = F.dropout(torch.relu(self.dense2(out)), self.dropout_rate)
        out = torch.relu(self.dense3(out))
        out = self.out_layer(out)
        # out = self.new_out(out)     
        return out

class SingleModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model_type = kwargs['model_type']
        self.mode = kwargs['mode']
        self.task = kwargs['task']
        model_type = self.model_type
        if model_type == 'RL':
            from model.RL_AR_Tree import RL_AR_Tree
            Encoder = RL_AR_Tree
        elif model_type == 'STG':
            from model.STGumbel_AR_Tree import STGumbel_AR_Tree
            Encoder = STGumbel_AR_Tree
        self.word_embedding = nn.Embedding(num_embeddings=kwargs['num_words'], embedding_dim=kwargs['word_dim'])                                 
        self.encoder = Encoder(**kwargs)
        self.classifier = Classifier(**kwargs)   # etrafından dolaşıyoruz, önemli değil, sıkıntı çıkabilir diye duruyor
        self.new_classifier = New_Classifier(**kwargs)
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()   # etrafından dolaşıyoruz, önemli değil, sıkıntı çıkabilir diye duruyor
        self.new_classifier.reset_parameters()

    def forward(self, ligand, length):
        words_embed = self.word_embedding(ligand)
        words_embed = self.dropout(words_embed)
        if self.model_type == 'STG':
            h, _, tree = self.encoder(words_embed, ligand, length)
            if self.mode == "emb":
                return h
            # logits = self.classifier(h)
            logits = self.new_classifier(h)
            supplements = {'tree': tree}        
        elif self.model_type == 'RL':
            h, _, tree, samples = self.encoder(words_embed, ligand, length)
            logits = self.classifier(h)
            supplements = {'tree': tree} 
            # samples prediction for REINFORCE
            sample_logits = self.classifier(samples['h'])
            supplements['sample_logits'] = sample_logits
            supplements['probs'] = samples['probs']
            supplements['sample_trees'] = samples['trees']
            supplements['sample_h'] = samples['h']
        return logits, supplements
