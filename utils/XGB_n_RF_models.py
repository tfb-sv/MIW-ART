##########################################################################################
# cd C:\Users\nural\OneDrive\Masaüstü\MARTree3_emb
# cd C:\Users\WS001\Desktop\MARTree3_emb
##########################################################################################
import pickle
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score, KFold

torch.manual_seed(0)
##########################################################################################
# TRAIN AND TEST BOTH THE XGBOOST AND RANFOM FOREST MODELS

df_names = ["bace", "bbbp2k", "tox21", "clintox"]   # , "hiv"
all_results = {}
seed = 0
for i in range(len(df_names)):
    scores, mean, std = {}, {}, {}
    ############################################## IMPORT DATASET
    df_name = df_names[i]
    ############################################## IMPORT TRAIN DATASET
    df_train_loc = "data/" + df_name + "_train.csv"
    df_train = pd.read_csv(df_train_loc)
    ############################################## IMPORT TEST DATASET
    df_test_loc = "data/" + df_name + "_test.csv"
    df_test = pd.read_csv(df_test_loc)
    ############################################## IMPORT ALL EMBBEDINGS (ALL = TRAIN + TEST)
    emb_loc = "e_out/" + df_name + "_emb.pkl"
    emb_file = open(emb_loc, 'rb')
    emb_all = pickle.load(emb_file)
    ############################################## CREATE X_TRAIN AND X_TEST from ALL EMBEDDINGS
    X_temp = emb_all["embedding"]
    X_all = []
    for i in range(len(X_temp)):
        vec = np.array(X_temp[i])
        X_all.append(vec)
    X_train = X_all[:len(df_train)]
    X_test = X_all[len(df_train):]
    ############################################## CREATE Y_TRAIN AND Y_TEST
    y_train = torch.tensor(df_train["affinity_score"])
    y_test = torch.tensor(df_test["affinity_score"])
    ############################################## SET KFOLD
    # kfold = KFold(n_splits=3, shuffle=True)
    ############################################## TRAIN XGB MODEL  
    xgb_model = XGBClassifier(learning_rate=0.03, 
                              n_estimators=600, 
                              objective='binary:logistic',
                              gpu_id=0,
                              random_state=seed,
                              )
    xgb_model.fit(X_train, y_train)
    ############################################## TEST XGB MODEL
    xgb_pred = xgb_model.predict(X_test)
    xgb_scores = roc_auc_score(y_test, xgb_pred)
    ##############################################
    # xgb_scores = cross_val_score(xgb_model, X_test, y_test,
    #                              cv=kfold, scoring="roc_auc")  
    ############################################## TRAIN RF MODEL
    rf_model = RandomForestClassifier(n_estimators=600, 
                                      random_state=seed,
                                      )
    rf_model.fit(X_train, y_train)
    ############################################## TEST RF MODEL
    rf_pred = rf_model.predict(X_test)
    rf_scores = roc_auc_score(y_test, rf_pred)
    ##############################################
    # rf_scores = cross_val_score(rf_model, X_test, y_test,
    #                             cv=kfold, scoring="roc_auc")
    ############################################## SAVE TEST RESULTS
    scores["xgb"] = np.round(xgb_scores, 3)
    scores["rf"] = np.round(rf_scores, 3)
    # mean["xgb"] = np.round(np.mean(xgb_scores), 3)
    # mean["rf"] = np.round(np.mean(rf_scores), 3)
    # std["xgb"] = np.round(np.std(xgb_scores), 3)
    # std["rf"] = np.round(np.std(rf_scores), 3)
    # all_results["df_name"] = [scores, mean, std]
    all_results["df_name"] = scores
    ############################################## PRINT TEST RESULTS
    print("\n\n>>  For", df_name.upper(), "dataset:\n", all_results["df_name"])

##########################################################################################







































