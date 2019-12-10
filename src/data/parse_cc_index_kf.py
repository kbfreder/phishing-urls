import gc
import gzip
import time
import json
import shutil
import os, sys
import tldextract
import collections
import pandas as pd
from tqdm import tqdm
import urllib.request
from multiprocessing import Pool
import random


storage_folder = '../data/raw/index_paths/'
remote_file_prefix = 'https://commoncrawl.s3.amazonaws.com/'

def read_every_line(fname, max_lines=-1):
    lines = []
    with open(fname, encoding='utf-8') as f:
        for i, l in enumerate(f):
            lines.append(l)
            if i>max_lines and max_lines>0:
                break
    return lines

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def save(url, filename):
    urllib.request.urlretrieve(url, filename, reporthook)

def list_multiprocessing(param_lst,
                         func,
                         **kwargs):

    workers = kwargs.pop('workers')

    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i,params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))

    # lists do not need such sorting, but this can be useful later
    result=sorted(result,key=lambda x:x[0])
    return [_[1] for _ in result]


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params,**kwargs)


def process_index_file_line(line):
    #
    assert type(line)==str

    try:
        lst = line.replace('\n','').split()
        ts = lst[1]
        data = json.loads(line.replace('\n','').split(ts)[-1].strip())
    except:
        return ()

    if data['status'] != '200':
        return ()
    else:
#         try:
#             language = data['languages']
#         except:
#             language = 'none'

        try:
            _tldextract = tldextract.extract(data['url'])
            tup = (ts,
                   data['url'],
                   _tldextract.suffix,
                   # data['length'],
                   # data['offset'],
                   # data['filename'],
#                    language
                )
            return tup
        except:
            return ()


def process_index_file(file_name):
    print('Unzipping index file ... ')

    df_name = file_name.replace('.gz','.feather')
    file_unzipped = file_name.split('.gz')[0]

    with gzip.open(file_name, 'rb') as f_in:
        with open(file_unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    lines = read_every_line(file_unzipped, 1e8)
    print('{} lines extracted'.format(len(lines)))

    print('Pre-processing index lines ... ')
    out = list_multiprocessing(lines, process_index_file_line, workers=4)
#     out = []

#     for line in lines:
#         out.append(process_index_file_line(line))

    # filter out blank lines
    out =  [_ for _ in out if _ != ()]

    print('Index pre-processed!')

    print('Processing index dataframe ... ')

    ts_list       = [_[0] for _ in out]
    url_list      = [_[1] for _ in out]
    tld           = [_[2] for _ in out]
    # length_list   = [_[3] for _ in out]
    # offset_list   = [_[4] for _ in out]
    # warc_list     = [_[5] for _ in out]
    # language_list = [_[6] for _ in out]

    cols = ['ts','url','tld']#,'length','offset','warc','language']
    df = pd.DataFrame(data={
        'ts':ts_list,
        'url':url_list,
        'tld':tld,
#         'length':length_list,
#         'offset':offset_list,
#         'warc':warc_list,
#         'language':language_list
    }
                      ,columns=cols)

#     df['wet'] = df.warc.apply(lambda x: x.replace('/warc/','/wet/').replace('.warc.','.warc.wet.'))
#     df['wet'] = df['wet'].apply(lambda x: file_prefix + x)


#     os.remove(file_name)
#     os.remove(file_unzipped)
#     print('Files removed ... ')

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print('Index dataframe is ready!')

    print('Saving Dataframe ... ')
    df.to_feather(df_name)
    print('Dataframe saved ... ')



def main():
    '''Assumes cc-index.paths.gz has already been downloaded to `storage_folder`.
        The `cc-index.paths` file contains paths to several hundred index files.
        Each compressed index file is ~775 MB, contains 7-10 million URLs,
        takes ~20 min to download, is ~6 GB once unzipped, and takes >100 min to
        extract and process, and results in a ~ 1 GB DataFrame of URLs & record IDs.
    '''
    index_file_name = 'cc-index.paths.gz'
    file_name = os.path.join(storage_folder, index_file_name)
    file_unzipped = file_name.split('.gz')[0]

    if not os.path.isfile(file_unzipped):
        with gzip.open(file_name, 'rb') as f_in:
            with open(file_unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    idx_lines = read_every_line(file_unzipped, 1e8) # aka cc_indexes
    num_lines = len(idx_lines)
    print('{} lines extracted'.format(num_lines))
    idx_lines = [line.replace('\n','') for line in idx_lines]

    i = random.choice(range(num_lines))
    idx_root = idx_lines[i]
    idx_file = idx_root.split('/')[-1]
    # file_dict[os.path.join(storage_folder, cc_index_file)] = file_prefix + cc_index
    local_file_path = os.path.join(storage_folder, idx_file)
    url = remote_file_prefix + idx_root

    start_time = time.time()
    print('PROCESSING INDEX FILE ...')
    print('Downloading index file {} ...'.format(local_file_path))
    save(url, local_file_path)
    print('Processing index file...')
    process_index_file(local_file_path)
    gc.collect()
    duration = int((time.time() - start_time) / 60)
    print('Elapsed time {} min'.format(duration))



if __name__ == "__main__":
    main()
