import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['lbl', 'msg']
df['lbl'] = df['lbl'].map({'spam': 1, 'ham': 0})

nltk.download('stopwords')
sw = set(stopwords.words('english'))

def clean(txt):
    txt = txt.lower()
    txt = re.sub(r'\d+', '', txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    return " ".join(w for w in txt.split() if w not in sw)

df['msg'] = df['msg'].apply(clean)

vec = TfidfVectorizer()
X = vec.fit_transform(df['msg'])
y = df['lbl']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "NB": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "RF": RandomForestClassifier()
}

for n, m in models.items():
    m.fit(X_tr, y_tr)
    y_p = m.predict(X_te)
    print(f"{n} Acc: {accuracy_score(y_te, y_p):.2f}")

with open('spam_model.pkl', 'wb') as f:
    pickle.dump((models["SVM"], vec), f)
