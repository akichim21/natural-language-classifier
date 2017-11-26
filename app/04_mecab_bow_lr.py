# This Python file uses the following encoding: utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from preprocessings.tokenizer import MeCabTokenizer
from preprocessings.livedoor import load_df

# 01との比較でjanomeをmecabに変えた

# mecabでトークナイズ
# BOWで文をベクトル化
# LogisticRegressionで分類

# mecabでトークナイズ
mecab = MeCabTokenizer()
def tokenize(word):
    tokens = [mecab.surface(token) for token in mecab.tokenize(word)]
    return " ".join(tokens)

# livedoorの記事をラベル付きでDataFrameとして読み込み
df = load_df()

# 文全てをtokenize
df['docs'] = df['docs'].apply(tokenize)

# BOWで文をベクトル化
count = CountVectorizer()
X_count = count.fit_transform(df['docs'].values)

# トレーニング:テスト = 8:2で分ける
X_train, X_test, Y_train, Y_test = train_test_split(X_count, df['labels'], test_size=0.2,random_state=3)

# LogisticRegressionで分類
clf = LogisticRegression()
clf.fit(X_train, Y_train)

# テストデータで正確性(accuracy)を表示
print(clf.score(X_test, Y_test))
