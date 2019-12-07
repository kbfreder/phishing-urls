import re
import gzip
import time
# just git clone https://github.com/erroneousboat/warc3.git
import warc
import nltk
import shutil
import os,sys
import tldextract
import pandas as pd
from tqdm import tqdm
import urllib.request

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
    
def read_every_line(fname,
                    max_lines=-1):
    lines = []
    with open(fname, encoding='utf-8') as f:
        for i, l in enumerate(f):
            lines.append(l)
            if i>max_lines and max_lines>0:
                break
    return lines

def remove_special_chars(text,char_list):
    for char in char_list:
        text=text.replace(char,'')
    return text.replace(u'\xa0', u' ')

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    return reg.sub('', string)

def process_web_text(text):
    # fist remove any remaining HTML
    text = remove_html_tags(text)
    # then split by line
    sentences = text.split('\n')
    # then omit sentences with more than 50% non printable chars
    sentences = [nltk.sent_tokenize(sentence) for sentence in sentences if len(sentence)//2<len(_remove_non_printed_chars(sentence))-2]
    sentences = [item for sublist in sentences for item in sublist]
    return sentences

def process_wet_file(file_name,
                     file_unzipped,
                     url_set):
    print('Unzipping index file ... ')
    
    df_name = file_name.replace('.warc.wet.gz','.feather')
    cols = ['url','domain','tld','sentence']
    df = pd.DataFrame(columns=cols)

    # unzip a file
    with gzip.open(file_name, 'rb') as f_in:
        with open(file_unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    lines = read_every_line(file_unzipped,
                            1e8)
    
    print('File unzipped ... ')
    
    print('Processing WET file ... ')

    with warc.open(file_unzipped) as f:
        for i,record in enumerate(f):
            if record.url in url_set:
                _tldextract = tldextract.extract(record.url)
                d = _tldextract.domain       
                tld = _tldextract.suffix  
                record.payload.read().decode("utf-8")
                sentences = process_web_text(text)
                
                temp_df = pd.DataFrame(data={
                    'url':[record.url]*len(sentences),
                    'domain':[d]*len(sentences),
                    'tld':[tld]*len(sentences),
                    'sentence':sentences}
                                       ,columns=cols)                 
                
                df = df.append(temp_df)

    print('WET file processed ... ')
    
    os.remove(file_name) 
    os.remove(file_unzipped) 

    print('Files removed ... ')
    
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df.to_feather(df_name)
    
    print('Df saved ... ')
    
storage_folder = 'data/'
file_prefix = 'https://commoncrawl.s3.amazonaws.com/'

# dummy code assuming that we will pre-process only the most popular file
dfs = []
for i in range(1,30):
    dfs.append(pd.read_feather(os.path.join(storage_folder,'cdx-000{}.feather'
                                            .format(str(i).zfill(2)))))
df = pd.concat(dfs)

url = df.wet.value_counts().index[0]
file_name = url.split('/')[-1]
file_unzipped = filename.replace('.warc.wet.gz','.warc')
url_set = set(df.url.unique())
save(url, filename)
process_wet_file(file_name,
                 file_unzipped,
                 url_set)    