'''Gets Phishing URLs
'''


import urllib.request
import json
import panda as pd

if __name__ == '__main__':
    # fetch URLs from given website
    url = 'https://openphish.com/feed.txt'
    txt = urllib.request.urlopen(url).read()
    lines = txt.decode().split('\n')
    df1 = pd.DataFrame(data=lines, columns=['url'])

    # fetch URLs from second website
    app_key = '1f171e316ef8a512369c33676ddfe160724d2d36146993892c70478f75f78691'
    url = 'http://data.phishtank.com/data/{}/online-valid.json'.format(app_key)

    with urllib.request.urlopen(url) as webpage:
        data = json.loads(webpage.read().decode())

    url_list = [x['url'] for x in data]
    df2 = pd.DataFrame(data=url_list, columns=['url'])

    # Merge df's
    df = pd.concat([df1, df2])
    df.drop_duplicates(inplace=True)

    # Save df
    df.to_pickle('../data/raw/phising_urls.pkl')
