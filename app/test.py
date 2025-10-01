import spacy
import re
from spacy.matcher import PhraseMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from train import sentences, labels

nlp = spacy.load("fr_core_news_md")

X_tr = np.array([nlp(s).vector for s in sentences])

def detect_intention_cos(message):
    test_vec = nlp(message).vector.reshape(1, -1)
    sims = cosine_similarity(X_tr, test_vec)
    idx = np.argmax(sims)
    return labels[idx]

print("\nje suis en terminale D et je veux des options d'études")
print(detect_intention_cos("je suis en terminale D et je veux des options d'études"))


def training():
    X = np.array([nlp(s).vector for s in sentences])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

modele = training()
print(modele.predict([nlp("je veux apprendre la couture").vector]))
