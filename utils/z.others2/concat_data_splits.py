

import pandas as pd

txt = "C:/Users/nural/OneDrive/Masaüstü/ART-Mol/data/lipo/"

task = "lipo"

train = pd.read_csv(f"{txt}{task}_train.csv")
valid = pd.read_csv(f"{txt}{task}_val.csv")
test = pd.read_csv(f"{txt}{task}_test.csv")

allx = pd.concat([train, valid, test], axis=0)

allx.to_csv(f"{txt}{task}_all.csv")