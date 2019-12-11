'''train_model.py

Trains RandomForestClassifier model on <input> file located in
 ../../data/processed. Saves model to ../../models.

Usage:
    train_model.py [options]

Options
    -i --input <file>       Filename, with extension, but without path
                            (../../data/processed). [default: train_df.feather]
    -o --output <file>      Filename for resulting model, without extension.
                            Will be save to ../../models/. [default: rf_model]
    -h --help               Show docstring.
    -t                      Test mode.

'''
import os
import sys
import time
import re
import feather
import joblib
# import pandas as pd
from docopt import docopt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.curdir)))
# project_dir = os.path.dirname(os.path.abspath(os.path.curdir))
new_path = os.path.join(project_dir, 'src')
sys.path.append(new_path)

import util as u
from model import pipeline as p
import model_config as cfg

clf_params = cfg.clf_params

def main(input_file, output_file, model_cols):
    start_time = time.time()

    print('Loading data...')
    input_path = os.path.join('../../data/processed/', input_file)
    df = feather.read_dataframe(input_path)

    print('Prepping data...')
    df = u.prep_for_model(df, model_cols)

    target = 'label'
    X = df.drop(columns=target)
    y = df[target]

    all_cols = df.columns

    # Preprocessing
    print('Getting pipeline ready...')
    proc_dict = {
    # keep code here for future, when we can use this feature
    #     'base_suffix':[p.Consolidate(1), OneHotEncoder(handle_unknown='ignore')]
                }

    num_cols = [col for col in all_cols if re.search('_cnt', col) is not None] + \
                ['length_url', 'hostname_entropy', 'url_entropy']

    bool_cols = [col for col in all_cols if re.search('_ind', col) is not None]

    pass_thru_cols = [col for col in all_cols if re.search('_frac_url_len', col) is not None]

    for col in num_cols:
        proc_dict[col] = [StandardScaler()]

    for col in bool_cols + pass_thru_cols:
        proc_dict[col] = [p.PassThrough()]

    # Pipeline
    preproc_pipe = FeatureUnion(p.gen_pipeline(model_cols, proc_dict))
    clf = RandomForestClassifier(**clf_params)
    pipe = make_pipeline(preproc_pipe, clf)

    # Fit & predict
    print('Fitting model....')
    pipe.fit(X, y)
    print('Predicting train...')
    y_pred = pipe.predict(X)
    score = f1_score(y, y_pred, pos_label='phishing')
    print('Training score: ', score)

    # Save model & predicions
    print('Saving model...')
    output_path = os.path.join('../../models/', output_file)
    joblib.dump(pipe, output_path)

    u.pickle_this(y_pred, '../../data/processed/train_pred.pkl')
    print('Script complete!')

    duration = (time.time() - start_time) / 60
    print('Elapsed time: {:.0f} minutes'.format(duration))


def test(input_file, output_file, model_cols):
    print('Running test...')
    # print(model_cols)
    clf = RandomForestClassifier(**clf_params)
    print(clf)


if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    input_file = arguments['--input']
    output_file = arguments['--output']
    test_mode = arguments['-t']
    model_cols = cfg.model_cols

    if test_mode:
        test(input_file, output_file, model_cols)
    else:
        main(input_file, output_file, model_cols)
