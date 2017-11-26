# This Python file uses the following encoding: utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from preprocessings.tokenizer import JanomeTokenizer
from preprocessings.stopwords import get_stopwords, remove_stopwords
from preprocessings.livedoor import load_df

# 01との比較で前処理でstopwordやstemmingをやった

# janomeでトークナイズ
# baseform_or_surfaceでstemming、exist_posで品詞に応じたstopword、頻出度に応じたstopwordを実施
# BOWで文をベクトル化
# LogisticRegressionで分類

# janomeでトークナイズ
# baseform_or_surfaceでstemming
# exist_posで品詞に応じたstopword
janome = JanomeTokenizer()
def tokenize(word):
    words = [janome.baseform_or_surface(token) for token in janome.tokenize(word) if janome.exist_pos(token)]
    return " ".join(words)

# livedoorの記事をラベル付きでDataFrameとして読み込み
df = load_df()


# 文全てをtokenize
df['docs'] = df['docs'].apply(tokenize)

# dlした既存のstopword, 頻出度に応じたstopwordを実施
stopwords = get_stopwords(df['docs'])

# BOWで文をベクトル化
count = CountVectorizer(stop_words = stopwords)
X = count.fit_transform(df['docs'].values)

# トレーニング:テスト = 8:2で分ける
X_train, X_test, Y_train, Y_test = train_test_split(X, df['labels'], test_size=0.2,random_state=3)

# LogisticRegressionで分類
clf = LogisticRegression()
clf.fit(X_train, Y_train)

# テストデータで正確性(accuracy)を表示
print(clf.score(X_test, Y_test))
