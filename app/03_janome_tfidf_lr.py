from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from preprocessings.tokenizer import JanomeTokenizer
from preprocessings.livedoor import load_df

# 01との比較はBOWをTFIDFに変えた

# janomeでトークナイズ
# TFIDFで文をベクトル化
# LogisticRegressionで分類

# janomeでトークナイズ
janome = JanomeTokenizer()
def tokenize(word):
    tokens = [janome.surface(token) for token in janome.tokenize(word)]
    return " ".join(tokens)

# livedoorの記事をラベル付きでDataFrameとして読み込み
df = load_df()

# 文全てをtokenize
df['docs'] = df['docs'].apply(tokenize)

# TFIDFで文をベクトル化
count = CountVectorizer()
X_count = count.fit_transform(df['docs'].values)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count)

# トレーニング:テスト = 8:2で分ける
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, df['labels'], test_size=0.2,random_state=3)

# LogisticRegressionで分類
clf = LogisticRegression()
clf.fit(X_train, Y_train)

# テストデータで正確性(accuracy)を表示
print(clf.score(X_test, Y_test))
