"""predict_model.py

Predict label on '../../data/processed/<input>' feather data using
'../../models/<model>' model. Note defaults below. Saves predictions
to a feather file format.

Usage:
    predict_model.py [options]

Options:
    -h --help                   Show doctsring.
    -i <file>, --input <file>   Input (test_df) filename with file extension but without
                                path (../../data/processed) [default: test_df.feather].
    -m <file>, --model <file>   Model filename without path (../../models/) or file extension
                                [default: rf_model]
    -o <file>, --output <file>  Output filename without path (../../data/processed/)
                                [default: test_pred.feather]
    -t --test                   Test mode. [default: False]
"""
import os
import sys
import logging
import time
import feather
import joblib
import pandas as pd
import numpy as np
import re
from docopt import docopt
from sklearn.metrics import f1_score

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.curdir)))
# project_dir = os.path.dirname(os.path.abspath(os.path.curdir))
new_path = os.path.join(project_dir, 'src')
sys.path.append(new_path)

import util as u
from model import pipeline as p

import model_config as cfg

def test(input_filename, model_filename, output_filename, model_cols):
    print('Running test')
    print(model_cols)


def main(input_filename, model_filename, output_filename, model_cols):
    start_time = time.time()

    # set-up
    input_path = '../../data/processed/' + input_filename
    model_path = '../../models/' + model_filename
    output_path = '../../data/processed/' + output_filename
    logging.basicConfig(level=logging.DEBUG)

    # load data & model
    logging.info('test data: {}'.format(input_path))
    logging.info('model: {}'.format(model_path))
    logging.debug('Loading data & model...')

    df = feather.read_dataframe(input_path)
    df = u.prep_for_model(df, model_cols)
    pipe = joblib.load(model_path)

    target = 'label'
    X = df.drop(columns=target)
    y = df[target]

    # predict
    logging.debug('Predicting...')
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)
    pp_cols = [c + '_PREDICT_PROBA' for c in pipe.classes_]
    y_proba_df = pd.DataFrame(data=y_proba, columns=pp_cols)

    score = f1_score(y, y_pred, pos_label='phishing')
    print('F1-score: ', score)

    logging.debug('Saving predictions...')
    # Make a df of the predictions
    pred_df = df.join(y_proba_df)
    pred_df['PREDICTION'] = y_pred
    pred_df['PRED_CORRECT_IND'] = np.where(pred_df['PREDICTION'] == pred_df['label'], 1, 0)


    # Save
    logging.debug('Saving...')
    feather.write_dataframe(pred_df, output_path)
    logging.info('Output (predictions) saved to: {}'.format(output_path))

    logging.info('Script complete!')
    duration = (time.time() - start_time) / 60
    print('Elapsed time: {:.0f} minutes'.format(duration))

if __name__ == '__main__':

    # load arguments
    arguments = docopt(__doc__, help=True)
    input_filename = arguments['--input']
    model_filename = arguments['--model']
    output_filename =  arguments['--output']
    test_mode = arguments['--test']
    model_cols = cfg.model_cols

    if test_mode:
        test(input_filename, model_filename, output_filename, model_cols)
    else:
        main(input_filename, model_filename, output_filename, model_cols)
