'''merge_split_data.py

Loads benign data from ../../data/raw/*.feather (assumes there is only
a single `.feather` file in this location). Loads phishing data from
../../data/raw/phishing_urls.pkl. Merges them together. Performs
train-test-split using 25% test size. Saves as train_df and test_df to
../../data/interim.

Usage:
    merge_split_data.py

Options:
    -h --help       Show docstring.

'''
import pandas as pd
import feather
import os, sys
import re
from sklearn.model_selection import train_test_split
import logging
import time
from docopt import docopt

input_folder = '../../data/raw/'
output_folder = '../../data/interim'
phishing_filename = 'phising_urls.pkl'

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    start_time = time.time()

    # Load Benign data
        # this assumes there is only one .feather file in the input_folder
    logging.debug('Loading benign data...')
    feather_file = [f for f in os.listdir(input_folder) if re.search('.feather', f) is not None][0]
    feather_path = os.path.join(input_folder, feather_file)
    benign_df = feather.read_dataframe(feather_path)
    benign_df.drop(columns=['ts'], inplace=True)
    benign_df['label'] = 'benign'

    # Load Phishing data
    logging.debug('Loading phishing data...')
    ph_df = pd.read_pickle(os.path.join(input_folder, phishing_filename))
    ph_df['label'] = 'phishing'

    # Merge data
    logging.debug('Merging data...')
    df = pd.concat([benign_df, ph_df])

    if len(df) != len(benign_df) + len(ph_df):
        logging.error("Merging of df's did not work")
        pass

    # Split into X, y; then train, test
    logging.debug('Performing train-test split...')
    X = df['url']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)

    # Re-combine X & y
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    # Save df's
    logging.debug('Saving data...')
    train_df.to_pickle(os.path.join(output_folder, 'train_df.pkl'))
    test_df.to_pickle(os.path.join(output_folder, 'test_df.pkl'))

    print('Script complete!')
    duration = (time.time() - start_time) / 60
    print('Elapsed time: {:.0f} minutes'.format(duration))
