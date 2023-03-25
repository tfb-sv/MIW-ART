cd C:\Users\nural\OneDrive\Masaüstü\MARTree

#########################################################################################################

import pandas as pd
import json
import numpy as np
from ete3 import Tree
import seaborn as sns
sns.set_theme(rc={'figure.figsize':(48, 27)}, font_scale=2.5, style="whitegrid")

#########################################################################################################

# TP = Total Point (Normalized with max length of related tree)
# TR = Total Repeat
# AL = Average Test Loss = AL / TR
# BP = Before Point = NT / TR
# FP = Final Point = BP * (1 - Average Test Loss)

#########################################################################################################
# FONKSİYONU TANIMLA

def token_inspect(df_name):  # pointization actually
    file_loc = "freq_dicts/freq_dict_ligand_" + df_name + ".json"
    with open(file_loc, "r") as f:
        raw_l = json.load(f)
    token_dict = {}
    for token in raw_l:
        total_point = 0
        total_repeat = 0
        total_loss = 0
        info = []
        for lig in raw_l[token]:   # kaç farklı kimyasalda tekrar ettiği
            smi_no = lig[0][0]
            max_len = lig[1]
            test_loss = lig[2]
            points = lig[3]
            for p in points:   # aynı kimyasalın içinde kaç kere tekrar ettiği
                total_point += (p / max_len)
                total_repeat += 1
            total_loss += test_loss
            info.append(smi_no)
        bp = total_point / total_repeat
        avg_loss = total_loss / len(raw_l[token])
        fp = bp * (1 - avg_loss)
        new_token = token.replace("{", "(")
        new_token = new_token.replace("}", ")")
        token_dict[new_token] = [fp, bp, avg_loss, info, total_point, total_repeat, len(info)]
    df_tokens = pd.DataFrame(token_dict).T
    df_tokens.columns = ["FP", "BP", "AL", "Info", "TP", "TR", "InfoL"]
    df_tokens = df_tokens.sort_values(by='InfoL', ascending=False)
    return df_tokens

#########################################################################################################
# SMI2CID IMPORT ET

with open("data/smi2cid.json", "r") as f:
    smi2cid = json.load(f)    
cid2smi = {v:k for k, v in smi2cid.items()}

#########################################################################################################
# ÖNEMLİ TOKENLARI BUL

my_thr = 200
point_thr = 0.9

imp_ones = {}
df_names = ["bace_cscaffold", "bbbp2kscaffold", "bbbp8kscaffold", "clintoxrandom", "hivscaffold", "tox21random"]
df_token_lst = []
for df_name in df_names:    
    imp_ones[df_name] = []
    df_tokens = token_inspect(df_name)
    df_token_lst.append(df_tokens)
    cnt = 0
    for i in range(len(df_tokens)):
        fp = df_tokens["FP"][i]
        avg_loss = df_tokens["AL"][i]
        info = df_tokens["Info"][i]
        if cnt > (my_thr - 1):
            break
        if fp >= point_thr:
            token = df_tokens.index[i]
            if "[OOV]" in token:
                continue
            temp2 = imp_ones[df_name]
            temp = []
            for smi_no in info:
                temp.append(smi_no)
            info = temp
            temp2.append([token, fp, info, avg_loss])
            imp_ones[df_name] = temp2
            cnt += 1

#########################################################################################################
# TEXT DOSYASININ İÇİNDEN GEREKLİ KISIMLARI BUL

with open("results_log.txt", "r") as f:
    viss = []
    for line in f:
        viss.append(line)          

fv = {}
for df_name in df_names:
    fv[df_name] = {}
    for lig in imp_ones[df_name]:
        # token = lig[0]
        # fp = lig[1]
        # avg_loss = lig[3]
        for smi_no in lig[2]:
            rgx1 = "Ligand " + str(smi_no)
            rgx2 = "=" * 10   # aslında 50 tane
            for i in range(len(viss)):
                if rgx1 in viss[i]:
                    start = i - 1
                    break
            for j in range(i + 1, len(viss)):                        
                if rgx2 in viss[j]:
                    end = j + 1             
                    break
            for k in range(start, end):
                viss_t = viss[k].replace("{", "(")
                viss[k] = viss_t.replace("}", ")")
            viss[start + 1] = ">> NRL" + str(smi_no) + " >> " + cid2smi[smi_no]
            fv[df_name][smi_no] = viss[start:end]
        
#########################################################################################################

# 1'den fazla ligand'da(yani unique) hem root'a yakın olma hem de test_loss değerinin düşük olması gözetildi.
# Final point > 0.95 olanlara odaklanıldı. (0-1 arasında bir puanlama bu.)
# Puanlama kimyasalların büyüklüğünden bağımsız hale getirildi.
    
#########################################################################################################
# YENİ TEXT DOSYALARINI OLUŞTUR

for df_name in df_names:
    for lig in imp_ones[df_name]:        
        token = lig[0]
        avg_loss = lig[3]
        output_name = "viss/" + df_name + "/" + token + ".txt"
        # fp = lig[1]
        with open(output_name, "w") as f:
            temp = "\n\tTOKEN:  " + token + "  >>  AVERAGE TEST LOSS:  " + str(np.round(avg_loss, 4)) + "\n\n"
            f.write(temp)
            for smi_no in lig[2]:
                for line in fv[df_name][smi_no]:                    
                    f.write(line)
#########################################################################################################

def passive_or_active(smiles, df_name):  # bace eklenmedi!
    if df_name == "bbbp2kscaffold":
        df_name2 = "bbbp2k.csv"
    elif df_name == "bbbp8kscaffold":
        df_name2 = "bbbp8k.csv"
    elif df_name == "clintoxrandom":
        df_name2 = "clintox.csv"
    elif df_name == "hivscaffold":
        df_name2 = "hiv.csv"
    elif df_name == "tox21random":
        df_name2 = "tox21.csv"
    elif df_name == "bace_cscaffold":
        df_name2 = "bace.csv"
    loc0 = "orj_data/" + df_name2
    df = pd.read_csv(loc0)
    acts = []
    for smi in smiles:
        if smi == "":
            continue    
        idx = df.index[df["smiles"] == smi].tolist()[0]
        activeness = df["affinity_score"][idx]
        acts.append(int(activeness))
    if (0 in acts) and (1 not in acts):
        result = "passive"
    elif (1 in acts) and (0 not in acts):
        result = "active"
    elif (1 in acts) and (0 in acts):
        result = "both"
    return result
            
#########################################################################################################
# ASU'NUN İNCELEMESİ İÇİN CSV FORMATINDA ÇIKTI AL

for act in ["active", "passive", "both"]:
    for df_name in df_names: # ESAS LİSTE
        asu_lst = []
        l_lst = []
        for lig in imp_ones[df_name]: 
            token = lig[0]
            smiles = []
            l = len(lig[2])   # kaç tane smiles var            
            for smi_no in lig[2]:
                smi = cid2smi[smi_no]
                smiles.append(smi)
            smis_activeness = passive_or_active(smiles, df_name)
            if smis_activeness == act:
                asu_lst.append([token, smiles])
                l_lst.append(l)
        ########################################################## BOŞLUKLA DOLDUR CSV'DEKİ MAX UZUNLUĞA KADARKİ YERİ
        if l_lst == []:
            continue
        max_l = max(l_lst)
        columns = ["Fragment"]
        for i in range(len(asu_lst)):
            l_add = max_l - len(asu_lst[i][1])
            if l_add < 0:
                print("problem", l_add, max_l)
                break
            for j in range(int(l_add)):
                asu_lst[i][1].append("")
        ########################################################## DÜZ LİSTE HALİNE GETİR
        asu_lst2 = []
        for i in range(len(asu_lst)):
            token = asu_lst[i][0]
            temp = [token]
            for j in range(len(asu_lst[i][1])):
                temp.append(asu_lst[i][1][j])
            asu_lst2.append(temp)
        ########################################################## SÜTUNLARI EKLE
        for i in range(max_l):
            temp = "Smiles" + str(i + 1)
            columns.append(temp)
        ########################################################## EXPORT ET
        asu_df = pd.DataFrame(asu_lst2)   
        asu_df.columns = columns   # ["Fragment", "Smiles"]
        asu_df.index += 1
        output_name = "asu_csvs/" + df_name + "_" + act + ".csv"
        asu_df.to_csv(output_name)

#########################################################################################################    
#########################################################################################################
#########################################################################################################
        















