############################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

############################################################

import deepchem as dc
import pandas as pd

############################################################

def get_molnet_from_deepchem(pre_task_name):
    ############################################################
    clms = ["smiles", "y_true"]
    ############################################################
    dct = {"bace": 0, "bbbp2k": 0, "clintox": -1, "tox21": -1, "lipo": 0}
    ############################################################
    if pre_task_name == "bace":
        task_name = "bace_classification"
    elif pre_task_name == "bbbp2k":
        task_name = "bbbp"
    else:
        task_name = pre_task_name
    ############################################################
    attr_name = f"load_{task_name}"
    func = getattr(dc.molnet, attr_name)
    _, (train, valid, test), _ = func()
    ############################################################
    train = pd.concat([pd.DataFrame(train.ids), pd.DataFrame(train.y[:,dct[pre_task_name]])], axis=1)
    train.columns = clms
    valid = pd.concat([pd.DataFrame(valid.ids), pd.DataFrame(valid.y[:,dct[pre_task_name]])], axis=1)
    valid.columns = clms
    test = pd.concat([pd.DataFrame(test.ids), pd.DataFrame(test.y[:,dct[pre_task_name]])], axis=1)
    test.columns = clms
    allx = pd.concat([train, valid, test], axis=0)
    ############################################################
    train.to_csv(f"data_new/{pre_task_name}/{pre_task_name}_train.csv")
    valid.to_csv(f"data_new/{pre_task_name}/{pre_task_name}_val.csv")
    test.to_csv(f"data_new/{pre_task_name}/{pre_task_name}_test.csv")
    allx.to_csv(f"data_new/{pre_task_name}/{pre_task_name}_all.csv")
    ############################################################
    return

############################################################

for ptn in ["bace", "bbbp2k", "clintox", "tox21", "lipo"]:
    get_molnet_from_deepchem(ptn)
    print(ptn)

############################################################




























