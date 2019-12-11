import time
import json
import shutil
import os
import sys
import tldextract
import collections
import pandas as pd
from tqdm import tqdm
import urllib.request
import zlib
import zlib
import json
import random
import requests
import util as u

def decompress_stream(stream):
    o = zlib.decompressobj(16 + zlib.MAX_WBITS)

    for chunk in stream:
        yield o.decompress(chunk)

    yield o.flush()


def extract_file(file_path):
    file_suffix = file_path.split('.')[-1]
    unzipped_path = file_path.replace('.' + file_suffix, '')
    
    if not os.path.exists(unzipped_path):
        with gzip.open(file_path, 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    return unzipped_path


# Set variables for Paths File and remote / web location
yearmonth = '2019-47'
url_prefix = 'https://commoncrawl.s3.amazonaws.com/'
paths_file_name = 'cc-index.paths.gz'

# Variables derives from the above
storage_folder = './crawl-data/CC-MAIN-' + yearmonth
paths_file = os.path.join(storage_folder, paths_file_name)
paths_file_unzipped = extract_file(paths_file)

# Read paths from paths file, convert to URLs
paths = u.read_every_line(paths_file_unzipped, 1e8)
print('{} lines extracted'.format(len(paths)))
path_urls = [url_prefix + path for path in paths]

# Access the index URLs...

    # Save index files to their own folder
index_folder = os.path.join(storage_folder, 'index-files/')

n_chunks = 700 # round up from estimate of 669
i = 0
for url in path_urls:
    # Derive save-to path from url
    save_file_name = url.split('/')[-1].replace('.gz', '')
    save_path = os.path.join(index_folder, save_file_name)

    # Open a stream Request to the URL, define generator
    r = requests.get(url, stream=True)
    data_stream = decompress_stream(r.iter_content(1024))

    # Create file, wrtie data chunks to it:
    save_file = open(save_path, 'wb')

    logging.debug('Downloading {}'.format(save_file_name))
    
    for i in range(n_chunks):
        if chunk:
            chunk = next(data_stream)
            save_file.write(chunk)
    
    # Close file & Request connection
    save_file.close()
    r.close()
    logging.debug('Download complete!')
    i += 1



# currently not used:
def parse_index_chunk(data, max_lines=-1):
    string_data = data.decode('UTF-8')  
    lines = string_data.split('\n')
    out = []

    i = 0
    for line in lines:
        if i > max_lines and max_lines > 0:
            break
        
        ts = line.split('{')[0].split()[-1] # timestamp
        line_json = '{' + line.split('{')[-1]
        line_data = json.loads(line_json)

        if line_data['status'] != '200':
            return ()
        else:
            # try:
            #     language = line_data['languages']
            # except:
            #     language = 'none'

            try:
                _tldextract = tldextract.extract(line_data['url'])
                tup = (ts,
                    line_data['url'],
                    _tldextract.suffix,
                    # line_data['length'],
                    # line_data['offset'],
                    # line_data['filename'],
                    # language              
                    )
            out.append(tup)
        except:
            return ()

        i += 1
    
    # remove blank lines
    out = [o for o in out if o != ()]
    return out