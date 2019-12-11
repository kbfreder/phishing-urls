'''make_features.py

Loads data from ../../data/interim/<filename> data. Creates features.
Saves as feather file format, with original filename stub (minus file
extension). Ex: train_df.pkl --> train_df.feather.

Usage:
    make_features.py <filename> [options]

Arguments:
    filename <file>       Filename, with extension. Data will be loaded
                          from ../../data/interim.

Options
    -h --help             Show docstring.
    -t                    Test mode.

'''

import re
from collections import Counter
import os
import pandas as pd
import numpy as np
import tldextract
import logging
from docopt import docopt
import time
import feather

logging.basicConfig(level=logging.DEBUG)


def shannon_specific_entropy(s):
    N = len(s)
    c = Counter(s)
    h_sum = 0
    for _, n_i in c.items():
        f_i = n_i/N
        h_sum += (f_i) * np.log2(f_i)

    h_sum = -h_sum
    return h_sum

def get_query(url):
    query_regex = re.compile(r"\?([a-z0-9\-._~%!$&'()*+,;=:@/]*)#?")
    try:
        return re.search(query_regex, url).group(0)
    except AttributeError:
        return None

def get_path(url):
    # note: this returns '/' if there is no path
    # source: https://www.oreilly.com/library/view/regular-expressions-cookbook/9780596802837/ch07s12.html
    path_regex = re.compile(r"^([a-z][a-z0-9+\-.]*:(//[^/?#]+)?)?([a-zA-Z0-9\-._~%!$&'()*+,;=:@/]*)")
    try:
        return re.findall(path_regex, url)[0][-1]
    except IndexError:
        return None

def main(filename):
    start_time = time.time()

    # Load data
    logging.debug('Loading data...')
    input_path = os.path.join('../../data/interim', filename)

    df = pd.read_pickle(input_path)

    # MAKE FEATURES
    logging.debug('Creating features...')
    # extract TLD parts
    df['protocol'] = df['url'].apply(lambda x: x.split('://')[0])
    df['tld_extract'] = df['url'].map(tldextract.extract)
    df['subdomain'] = df['tld_extract'].apply(lambda x: x.subdomain)
    df['domain'] = df['tld_extract'].apply(lambda x: x.domain)
    df['suffix'] = df['tld_extract'].apply(lambda x: x.suffix)
    df['hostname'] = df['tld_extract'].apply(lambda x: '.'.join(x))
    df['path'] = df['url'].apply(get_path)
    df['query'] = df['url'].apply(get_query)

    # Subdomain ind's
    df['subdomain_null_ind'] = np.where(df['subdomain'] == '', 1, 0)
    df['subdomain_www_ind'] = np.where(df['subdomain'] == 'www', 1, 0)

    # String lengths
    df['length_url'] = df['url'].map(len)
    df['length_domain'] = df['domain'].map(len)
    df['length_path'] = df['path'].map(len)

    # "Special" characters: counts, indicators
    df['domain_dot_cnt'] = df['domain'].apply(lambda s: s.count('.'))
    df['url_slash_cnt'] = df['url'].apply(lambda x: x.count('/'))
    df['path_dot_cnt'] = df['path'].apply(lambda x: x.count('.'))
    df['hostname_dash_cnt'] = df['hostname'].apply(lambda x: x.count('-'))

    digits = re.compile(r'[0-9]')
    df['url_digit_cnt'] = df['url'].apply(lambda x: len(re.findall(digits, x)))

    special_chars = re.compile(r"[$-_.+!*'\(\)\,]")
    df['url_special_char_cnt'] = df['url'].apply(lambda x: len(re.findall(special_chars, x)))

    reserved_chars = re.compile(r'[;/\?:@=&]')
    df['url_reserved_char_cnt'] = df['url'].apply(lambda x: len(re.findall(reserved_chars, x)))

    hex_pattern = re.compile(r"(%[0-9A-F]{2})")
    df['url_hex_pattern_ind'] = df['url'].apply(lambda x:
                                                1 if re.search(hex_pattern, x)
                                                is not None else 0)

    # Entropy
    df['hostname_entropy'] = df['hostname'].apply(shannon_specific_entropy)
    df['url_entropy'] = df['url'].apply(shannon_specific_entropy)

    # Suspicious words
    suspicious_words = ['php', 'abuse', 'admin', 'verification']

    for word in suspicious_words:
        col_name = word + '_ind'
        df[col_name] = np.where(df['url'].str.count(word) == 0, 0, 1)

    cols_to_convert = ['length_path', 'length_domain', 'url_slash_cnt',
                       'url_digit_cnt', 'url_special_char_cnt',
                       'url_reserved_char_cnt']

    for col in cols_to_convert:
        new_col_name = col + '_frac_url_len'
        df[new_col_name] = df[col] / df['length_url']

    # Save data
        # pickling does not work. Use feather instead
    logging.debug('Saving data...')
    df.drop(columns=['tld_extract'], inplace=True)
    output_path = os.path.join('../../data/processed', filename.replace('.pkl', '.feather'))
    feather.write_dataframe(df, output_path)

    print('Script complete!')

    duration = (time.time() - start_time) / 60
    print('Elapsed time: {:.0f} minutes'.format(duration))

def test(filename):
    print('Running test...')
    print(filename)

if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    filename = arguments['<filename>']
    test_mode = arguments['-t']

    if test_mode:
        test(filename)
    else:
        main(filename)
