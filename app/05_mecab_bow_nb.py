# This Python file uses the following encoding: utf-8
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from preprocessings.tokenizer import MeCabTokenizer
from preprocessings.livedoor import load_df

# 01との比較でロジスティック回帰(Logisticregression)を多項分布ナイーブベイズ分析(MultinomialNB)
# janomeからmecabに変更

# mecabでトークナイズ
# BOWで文をベクトル化
# 多項分布ナイーブベイズ分析(MultinomialNB)で分類

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

# MultinomialNBで分類
nb = MultinomialNB()
nb.fit(X_train, Y_train)

# テストデータで正確性(accuracy)を表示
print(nb.score(X_test, Y_test))
