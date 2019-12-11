'''nlp_features.py

Make NLP features from <filename> data. Loads from ../../data/interim.
Saves sparse array to ../../processed

Usage:
    nlp_features.py <filename> [options]

Arguments:
    filename <file>       Filename, with extension. Data will be loaded
                            from ../../data/interim.

Options
    -h --help               Show docstring.
    -t                      Test mode.

'''
import os
import logging
import pickle
import pandas as pd
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import feather

# Load data
def main(filename):
    file_path = os.path.join('../../data/interim', filename)
    save_path = '../../data/processed/'

    # Load data
    logging.debug('Loading data...')
    print('Loading data...')
    file_ext = file_path.split('.')[-1]

    if file_ext in ('pkl', 'pickle'):
        df = pd.read_pickle(file_path)
    elif file_ext == 'feather':
        feather.read_dataframe(file_path)
    else:
        print('File format not recognized.')
        return

    url_words = set('com', 'net', 'gov', 'edu', 'http', 'https', 'www')
    stopwords = set(ENGLISH_STOP_WORDS).union(url_words)
    # The token pattern excludes "words" that start with a digit or
    # an underscore (_).
    vectorizer = CountVectorizer(ngram_range=(1, 1),
                                 token_pattern='(?u)\\b[a-zA-Z]\\w+\\b',
                                 stop_words=stopwords)
    tfidf_output = vectorizer.fit_transform(df['path'])
    tfidf_features = vectorizer.get_feature_names()
    print('Total number of features: {}'.format(len(tfidf_features)))

    #Save data
    filestub = filename.split('.pkl')[0]
    output_filename = 'tfidf_output_' + filestub + '.feather'
    feather.write_dataframe(tfidf_output, os.path.join(save_path, output_filename))

    features_filename = 'tfidf_features_' + filestub + '.txt'
    with open(os.path.join(save_path, features_filename), 'wb') as picklefile:
        pickle.dump(tfidf_features, picklefile)

    # to open:
    # with open (os.path.join(save_path, features_filename), 'rb') as picklefile:
        # features = pickle.load(picklefile)


def test(filename):
    print('Running test...')
    print(filename)

if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    filename = arguments['<filename>']
    test_mode = arguments['-t']

    logging.basicConfig(level=logging.DEBUG)

    if test_mode:
        test(filename)
    else:
        main(filename)
