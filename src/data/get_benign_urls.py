'''get_benign_urls.py

Downloads URL data from cc-index paths

Usage:
    get_benign_urls.py download <n-chunks> [options]
    get_benign_urls.py parse [options]
    get_benign_urls.py download <n-chunks> parse
    get_benign_urls.py [options]

Options
    -v                      Verbose
    -h --help               Show docstring.
    -t                      Test mode.

'''

import time
import json
import shutil
import os
import sys
import logging

import pandas as pd
import feather
import zlib
import json
import random
import requests
from pandas.io.json import json_normalize
from docopt import docopt


n_chunks = 1500 # round up from estimate of 669
               # this yielded ~900,000 URLs
               # goal was 3,000,000
               # so, need to triple

# Set variables for Paths File and remote / web location
yearmonth = '2019-47'
url_prefix = 'https://commoncrawl.s3.amazonaws.com/'
paths_file_name = 'cc-index.paths.gz'
storage_folder = '../../data/raw/'


def extract_file(file_path):
    file_suffix = file_path.split('.')[-1]
    unzipped_path = file_path.replace('.' + file_suffix, '')

    if not os.path.exists(unzipped_path):
        with gzip.open(file_path, 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return unzipped_path


def read_every_line(fname, max_lines=-1):
    lines = []
    with open(fname, encoding='utf-8') as f:
        for i, line in enumerate(f):
            lines.append(line.replace('\n',''))
            if i > max_lines and max_lines > 0:
                break
    return lines


def decompress_stream(stream):
    o = zlib.decompressobj(16 + zlib.MAX_WBITS)

    for chunk in stream:
        yield o.decompress(chunk)

    yield o.flush()




# Variables derived from the above
# storage_folder = './crawl-data/CC-MAIN-' + yearmonth
paths_file = os.path.join(storage_folder, paths_file_name)
paths_file_unzipped = extract_file(paths_file)
index_folder = os.path.join(storage_folder, 'index-files/')

if not os.path.isdir(index_folder):
    os.mkdir(index_folder)


def download_index_data(n_chunks):
    # Read paths from paths file, convert to URLs
    paths = read_every_line(paths_file_unzipped, 1e8)
    path_urls = [url_prefix + path for path in paths]
    print('{} paths extracted from path file'.format(len(paths)))

    # Download chunks of data from each index url
    i = 0
    start_time = time.time()
    for url in path_urls:
        # Derive save-to path from url
        save_file_name = url.split('/')[-1].replace('.gz', '')

        if 'cdx-' in save_file_name:
            save_path = os.path.join(index_folder, save_file_name)

            # Open a stream Request to the URL, define generator
            r = requests.get(url, stream=True)
            data_stream = decompress_stream(r.iter_content(1024))

            # Create file, wrtie data chunks to it:
            save_file = open(save_path, 'wb')

            # print('Downloading {}'.format(save_file_name))
            if i+1 % 10 == 0 or i+1 == 1:
                sys.stdout.write('\rDownloading data from index {} out of {}'.format(i, len(path_urls)))
                sys.stdout.flush()

            for _ in range(n_chunks):
                chunk = next(data_stream)
                if chunk:
                    save_file.write(chunk)

            # Close file & Request connection
            save_file.close()
            r.close()
        i += 1

    print('All downloads complete!')
    duration = (time.time() - start_time) / 60
    print('Elapsed time: {:.0f} minutes'.format(duration))
    print(' ')



# Parse the index files


def _parse_index_file(file_name, cols_to_keep):
    # Read data from file
    with open(file_name, 'rb') as f:
        data = f.read()

    # Decode data
    string_data = data.decode('UTF-8')

    # Parse data
    lines = string_data.split('\n')
    file_data = []

    for line in lines:
        if line:
            ts = line.split('{')[0].split()[-1] # timestamp
            line_json = '{' + line.split('{')[-1]
            try:
                line_data = json.loads(line_json)
                line_data.update({'ts': ts})
            except:
                logging.info('Error parsing json data {} in file {}'.format(line_json, file_name))
                line_data = {}

        file_data.append(line_data)

    df = json_normalize(file_data)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=[col for col in df.columns if col not in cols_to_keep])

    return df


def parse_files(cols_to_keep=['ts', 'url', 'languages']):
    i = 1
    master_df = pd.DataFrame()

    for file in os.listdir(index_folder):
        if 'cdx-' in file:
            if i % 10 == 0 or i == 1:
                sys.stdout.write('\rParsing file {} out of {}'.format(i, len(os.listdir(index_folder))))
                sys.stdout.flush()

            file_path = os.path.join(index_folder, file)
            df = _parse_index_file(file_path, cols_to_keep)
            master_df = pd.concat([master_df, df], sort=False)
            i += 1

    print('Parsing complete! {} total records extracted.'.format(len(master_df)))

    logging.debug('Saving DataFrame...')
    data_path = '../data/raw/'
    master_file = 'cc_urls_' + yearmonth
    feather.write_dataframe(master_df, os.path.join(data_path, master_file))

if __name__ == '__main__':
    arguments = docopt(__doc__, help=True)
    logging.basicConfig(level=logging.INFO)

    if arguments['-t']:
        print('Test mode!')

    if arguments['download']:
        download_index_data(int(arguments['<n-chunks>']))

    if arguments['parse']:
        parse_files()
