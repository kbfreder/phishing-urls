import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import make_scorer, classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import log_loss, precision_recall_curve

import itertools
import re
import pickle

import matplotlib.pyplot as plt


def pickle_this(this, filepath):
    with open (filepath, 'wb') as picklefile:
        pickle.dump(this, picklefile)

def open_pickle(filepath):
    with open (filepath, 'rb') as picklefile:
        this = pickle.load(picklefile)
        return this

seed = 19


def read_every_line(fname, max_lines=-1):
    lines = []
    
    with open(fname, encoding='utf-8') as f:
        for i, line in enumerate(f):
            lines.append(line.replace('\n',''))
            if i > max_lines and max_lines > 0:
                break

    return lines



def assess_model_df(preprocessor, model, X, y, n=5):
    '''Input X is a dataframe. Performs stratified k-fold cross-validation,
    returns ALL THE THINGS:
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
        pipe.fit(X.loc[train], y[train])
        # pipe.fit(X[train], y[train])
        y_pred = pipe.predict(X.loc[test])
        # y_pred = pipe.predict(X[test])
        # CatBoost's predict output is float64. sklearn scoring metrics require integers
        y_pred = y_pred.astype(int)
        y_proba = pipe.predict_proba(X.loc[test])[:,1]
        # y_proba = pipe.predict_proba(X[test])[:,1]

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


def assess_model(preprocessor, model, X, y, n=5):
    '''Input X is an np.array. Performs stratified k-fold cross-validation,
    returns ALL THE THINGS:
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
    pipe.fit(X.loc[train], y[train])
    # pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X.loc[test])
    # y_pred = pipe.predict(X_test)
    # CatBoost's predict output is float64. sklearn scoring metrics require integers
    y_pred = y_pred.astype(int)
    y_proba = pipe.predict_proba(X.loc[test])[:,1]
    # y_proba = pipe.predict_proba(X_test)[:,1]

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
    """Prepares dataframe `df` for modeling.
        - Converts int datatype to np.float64
        - Drops na's from `model_cols`
    """

    int_cols = df.select_dtypes(include='int').columns
    int_cols = [col for col in int_cols if re.search('_ind', col) is None]

    for col in int_cols:
        df[col] = df[col].astype(np.float64)

    # moved this to 'make_features' - should no longer need here
    # cols_to_convert = ['length_path', 'length_domain', 'url_slash_cnt',
    #                     'url_digit_cnt', 'url_special_char_cnt',
    #                     'url_reserved_char_cnt']
    #
    # for col in cols_to_convert:
    #     new_col_name = col + '_frac_url_len'
    #     df[new_col_name] = df[col] / df['length_url']

    # TODO: add a before / after length check
    df = df.dropna(subset=model_cols)

    return df


def plot_conf_matrix(conf_mat, labels):
    '''Plots confusion matrix conf_mat, labeling with labels [list]'''
    fmt = '{:,}'
    thresh = conf_mat.max() / 2.
    classes = labels

    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, s=fmt.format(conf_mat[i, j]),
                     horizontalalignment="center",
                     size='16',
                     color="white" if conf_mat[i, j] > thresh else "black")

    plt.show()


def plot_prec_recall_curve(y_true, y_proba, pos_label=1):
    prec, rec, thr = precision_recall_curve(y_true, y_proba, pos_label='phishing')
    thr_adj = np.append(thr, 1)

    fig = plt.figure(figsize=(5.3,4))
    ax = fig.add_subplot(111)

    # plt.plot(rec, prec, label="", color='#56939E', lw=3, alpha=.8)
    plt.plot(rec, prec, label="", color='#70ADBA', lw=3, alpha=.8)


    plt.xlim([-0.0, 1.05])
    plt.ylim([-0.0, 1.05])
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],labels=["0.2", "0.4","0.6","0.8","1.0"],fontsize=14)

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

    return prec, rec, thr_adj
