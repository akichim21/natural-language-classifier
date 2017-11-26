# This Python file uses the following encoding: utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
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

def tokenize_by_pos(word):
    tokens = [mecab.surface(token) for token in mecab.tokenize(word) if mecab.exist_pos(token, ('動詞', '形容詞', '形容動詞', '名詞'))]
    return " ".join(tokens)

# livedoorの記事をラベル付きでDataFrameとして読み込み
df = load_df()

# トレーニング:テスト = 9:1で分ける
X_train, X_test, Y_train, Y_test = train_test_split(df['docs'], df['labels'], test_size=0.1,random_state=3)

grid_list = [
    # CountVectorizer, LogisticRegressionを検証
    # CountVectorizer: tokenizer
    # Logisticregression: penalty, C
    {
        'param_grid': {
            'vect__ngram_range': [(1, 1)],
            'vect__tokenizer': [tokenize, tokenize_by_pos],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 50.0, 100.0]
        },
        'pipeline': Pipeline([
            ('vect', CountVectorizer()),
            ('clf', LogisticRegression())
        ])
    },
    # TfidfVectorizer, LogisticRegressionを検証
    # TfidfVectorizer: tokenizer, use_idf
    # Logisticregression: penalty, C
    {
        'param_grid': {
            'vect__ngram_range': [(1, 1)],
            'vect__tokenizer': [tokenize, tokenize_by_pos],
            'vect__use_idf': [True, False],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 50.0, 100.0]
        },
        'pipeline': Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
    },
]
for grid in grid_list:
    gs = GridSearchCV(grid["pipeline"], grid["param_grid"], n_jobs = -1, cv = 10, scoring = 'accuracy', verbose = 1)
    gs.fit(X_train, Y_train)
    print('Best parameter set: %s' % gs.best_params_)
    print('CV Accuracy: %.3f' % gs.best_score_)
    clf = gs.best_estimator_
    print('Test CV Accuracy: %.3f' % clf.score(X_test, Y_test))

