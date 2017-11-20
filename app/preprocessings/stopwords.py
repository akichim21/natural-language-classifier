import os
import urllib.request
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STOPWORDS_PATH = os.path.join(DATA_DIR, 'dl_stopwords.txt')

def download():
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if not os.path.exists(STOPWORDS_PATH):
        print('start downloading')
        urllib.request.urlretrieve(url, STOPWORDS_PATH)

def get_stopwords(docs, n=100, min_freq=1):
    fdist = Counter()
    for doc in docs:
        for word in doc.split(' '):
            fdist[word] += 1
    common_words = { word for word, freq in fdist.most_common(n) }
    rare_words = { word for word, freq in fdist.items() if freq <= min_freq }

    download()
    fo = open(STOPWORDS_PATH, 'r')
    dl_words = { line.strip() for line in fo.read().split('\n') if not line == '' }
    fo.close()

    stopwords = list(common_words.union(rare_words).union(dl_words))
    print('{}/{}'.format(len(stopwords), len(fdist)))
    return stopwords

def remove_stopwords(words, stopwords):
    words = [word for word in words if word not in stopwords]
    return words
