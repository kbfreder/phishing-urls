import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import make_scorer, classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import log_loss

import pickle

def pickle_this(this, filepath):
    with open (filepath, 'wb') as picklefile:
        pickle.dump(this, picklefile)

def open_pickle(filepath):
    with open (filepath, 'rb') as picklefile:
        this = pickle.load(picklefile)
        return this

seed = 19

def assess_model(preprocessor, model, X, y, n=5):
    '''Stratified k-fold cross-validation, returns ALL THE THINGS:
    precision, recall, f1-score, confusion matrix, AUC, accuracy'''
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)

    precs = []
    recalls = []
    f1s = []
    conf_mat = []
    aucs = []
    accs = []

    for train, test in cv.split(X, y):
        # pipe.fit(X.loc[train], y[train])
        pipe.fit(X[train], y[train])
        # y_pred = pipe.predict(X.loc[test])
        y_pred = pipe.predict(X[test])
        # CatBoost's predict output is float64. sklearn scoring metrics require integers
        y_pred = y_pred.astype(int)
        # y_proba = pipe.predict_proba(X.loc[test])[:,1]
        y_proba = pipe.predict_proba(X[test])[:,1]

        precs.append(precision_score(y[test],y_pred, average=None))
        recalls.append(recall_score(y[test], y_pred, average=None))
        f1s.append(f1_score(y[test], y_pred, average=None))
        conf_mat.append(confusion_matrix(y[test], y_pred))
        aucs.append(roc_auc_score(y[test],y_proba))
        accs.append(accuracy_score(y[test], y_pred))

    # print(f1s, conf_mat)
    twos = [precs, recalls, f1s]
    data1 = [[s[i][j] for i in range(n)] for j in range(2) for s in twos]
    data1_new = np.mean(data1, axis=1).reshape(1,6)
    df1 = pd.DataFrame(data1_new,
             columns=['Precision-0', 'Recall-0 (Specificty)','F1score-0','Precision-1',
             'Recall-1 (Sensitivity)','F1score-1']).mean()
    data2 = [[s[i][j] for j in range(2) for i in range(2)] for s in conf_mat]
    df2 = pd.DataFrame(data2, columns=['TN','FN','FP','TP']).mean()
    df3 = pd.DataFrame(list(zip(aucs, accs)), columns=['AUC','Accuracy']).mean()
    return df1.append(df2.append(df3))


def assess_model_only(model, X, y, n=5):
    '''Stratified k-fold cross-validation, returns ALL THE THINGS:
    precision, recall, f1-score, confusion matrix, AUC, accuracy'''
    pipe =  Pipeline(steps=[('classifier', model)])
    cv = StratifiedKFold(n_splits=n, random_state=seed)

    precs = []
    recalls = []
    f1s = []
    conf_mat = []
    aucs = []
    accs = []

    for train, test in cv.split(X, y):
        pipe.fit(X[train], y[train])
        y_pred = pipe.predict(X[test])
        # CatBoost's predict output is float64. sklearn scoring metrics require integers
        y_pred = y_pred.astype(int)
        y_proba = pipe.predict_proba(X[test])[:,1]

        precs.append(precision_score(y[test],y_pred, average=None))
        recalls.append(recall_score(y[test], y_pred, average=None))
        f1s.append(f1_score(y[test], y_pred, average=None))
        conf_mat.append(confusion_matrix(y[test], y_pred))
        aucs.append(roc_auc_score(y[test],y_proba))
        accs.append(accuracy_score(y[test], y_pred))

    # print(f1s, conf_mat)
    twos = [precs, recalls, f1s]
    data1 = [[s[i][j] for i in range(n)] for j in range(2) for s in twos]
    data1_new = np.mean(data1, axis=1).reshape(1,6)
    df1 = pd.DataFrame(data1_new,
             columns=['Precision-0', 'Recall-0 (Specificty)','F1score-0','Precision-1',
             'Recall-1 (Sensitivity)','F1score-1']).mean()
    data2 = [[s[i][j] for j in range(2) for i in range(2)] for s in conf_mat]
    df2 = pd.DataFrame(data2, columns=['TN','FN','FP','TP']).mean()
    df3 = pd.DataFrame(list(zip(aucs, accs)), columns=['AUC','Accuracy']).mean()
    return df1.append(df2.append(df3))

def assess_model_no_cv(preprocessor, model, X_train, y_train, X_test, y_test):
    '''Returns ALL THE THINGS:
    precision, recall, f1-score, confusion matrix, AUC, accuracy'''
    n=1
    pipe =  Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    # cv = StratifiedKFold(n_splits=n, random_state=seed)

    precs = []
    recalls = []
    f1s = []
    conf_mat = []
    aucs = []
    accs = []

    # for train, test in cv.split(X, y):
    # pipe.fit(X.loc[train], y[train])
    pipe.fit(X_train, y_train)
    # y_pred = pipe.predict(X.loc[test])
    y_pred = pipe.predict(X_test)
    # CatBoost's predict output is float64. sklearn scoring metrics require integers
    y_pred = y_pred.astype(int)
    # y_proba = pipe.predict_proba(X.loc[test])[:,1]
    y_proba = pipe.predict_proba(X_test)[:,1]

    precs.append(precision_score(y_test,y_pred, average=None))
    recalls.append(recall_score(y_test, y_pred, average=None))
    f1s.append(f1_score(y_test, y_pred, average=None))
    conf_mat.append(confusion_matrix(y_test, y_pred))
    aucs.append(roc_auc_score(y_test,y_proba))
    accs.append(accuracy_score(y_test, y_pred))

    # print(f1s, conf_mat)
    twos = [precs, recalls, f1s]
    data1 = [[s[i][j] for i in range(n)] for j in range(2) for s in twos]
    data1_new = np.mean(data1, axis=1).reshape(1,6)
    df1 = pd.DataFrame(data1_new,
             columns=['Precision-0', 'Recall-0 (Specificty)','F1score-0','Precision-1',
             'Recall-1 (Sensitivity)','F1score-1']).mean()
    data2 = [[s[i][j] for j in range(2) for i in range(2)] for s in conf_mat]
    df2 = pd.DataFrame(data2, columns=['TN','FN','FP','TP']).mean()
    df3 = pd.DataFrame(list(zip(aucs, accs)), columns=['AUC','Accuracy']).mean()
    return df1.append(df2.append(df3))


def prep_for_model(df, model_cols):
    '''Prepares dataframe `df` for modeling.
    `model_cols` are the columns from which to drop na's
    '''

    int_cols = df.select_dtypes(include='int').columns
    int_cols = [col for col in int_cols if re.search('_ind', col) is None]

    for col in int_cols:
        df[col] = df[col].astype(np.float64)

    cols_to_convert = ['length_path', 'length_domain', 'url_slash_cnt',
                        'url_digit_cnt', 'url_special_char_cnt',
                        'url_reserved_char_cnt']

    for col in cols_to_convert:
        new_col_name = col + '_frac_url_len'
        df[new_col_name] = df[col] / df['length_url']

    df = df.dropna(subset=model_cols)

    return df
