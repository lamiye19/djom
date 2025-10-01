import spacy
import re
from spacy.matcher import PhraseMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from app.train import sentences, labels

nlp = spacy.load("fr_core_news_md")



X_train = np.array([nlp(s).vector for s in sentences])

def detect_intention_cos(message):
    test_vec = nlp(message).vector.reshape(1, -1)
    sims = cosine_similarity(X_train, test_vec)
    idx = np.argmax(sims)
    return labels[idx]

print("\nje suis en terminale D et je veux des options d'études")
print(detect_intention_cos("je suis en terminale D et je veux des options d'études"))


def train():
    X = np.array([nlp(s).vector for s in sentences])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

modele = train()
print(modele.predict([nlp("je veux apprendre la couture").vector]))



patterns = {
    "info_filiere": [
        r"c[' ]?est quoi\b",
        r"qu[' ]?est-ce qu[' ]?on apprend\b",
        r"définition\b",
        r"présentation\b",
        r"infos? sur\b",
        r"expliquer\b",
        r"combien coûte\b",
        r"coût\b",
        r"frais d[' e]? (scolarité|formation|admission)\b",
    ],
    "info_metier": [
        r"que fait\b",
        r"en quoi consiste\b",
        r"mission(s)? d[' ]?un\b",
        r"travail(le)? de\b",
        r"description du poste\b"
    ],
    "debouches": [
        r"débouchés?\b",
        r"opportunités?\b",
        r"après (un|le|la)\b",
        r"qu[' ]?est-ce qu[' ]?on peut faire après\b",
        r"carrière(s)? possible(s)?\b",
        r"emplois? accessibles?\b"
    ],
    "conditions_acces": [
        r"matières?\b",
        r"faut[- ]?il\b",
        r"conditions?\b",
        r"prérequis\b",
        r"quelle (scolarité|formation) pour\b",
        r"niveau requis\b"
    ],
    "comparaison": [
        r"différence entre\b",
        r"comparé à\b",
        r"mieux que\b",
        r"vs\b",
        r"entre\b",
        r"(par rapport à)\b"
    ],
    "orientation_perso": [
        r"je veux\b",
        r"j’aime\b",
        r"qu[' ]?est-ce que tu me conseilles\b",
        r"quelle filière pour moi\b",
        r"je voudrais\b",
        r"si j’aime (.+) quelle orientation\b"
    ]
}

# Extraction d’entités
def detect_intention(question):
    question_lower = question.lower()
    scores = {intent: 0 for intent in patterns}

    for intent, regex_list in patterns.items():
        for reg in regex_list:
            if re.search(reg, question_lower):
                scores[intent] += 1

    # Trier par score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Filtrer selon le seuil
    threshold = 1
    valid_intents = [(i, s) for i, s in sorted_scores if s >= threshold]

    return valid_intents


# Extraction simple des entités (à enrichir avec des listes ou NER custom)
regex_diplomes = re.compile(
    r"\b(?:bac(?:\s+(?:d|c|a\d|scientifique|litt(?:éraire)?))?|"
    r"(?:bts|dut)|"
    r"licence(?:\s+(?:pro(?:fessionnelle)?|recherche|en\s+[a-zàâçéèêëîïôûùüÿñæœ\s]+)?)?|"
    r"master(?:\s+(?:1|2|pro(?:fessionnel)?|recherche|en\s+[a-zàâçéèêëîïôûùüÿñæœ\s]+)?)?|"
    r"doctorat(?:\s+(?:en\s+[a-zàâçéèêëîïôûùüÿñæœ\s]+)?)?)\b",
    re.IGNORECASE
)


regex_metiers = re.compile(
    r"\b(ingénieur|médecin|juge|avocat|enseignant|développeur|data scientist|architecte)\b",
    re.IGNORECASE
)

regex_matieres = re.compile(
    r"\b(math(ématiques)?|physique|chimie|biologie|philosophie|histoire|géographie|anglais|français)\b",
    re.IGNORECASE
)

regex_lieux = re.compile(
    r"\b(togo|lomé|kara|sokode|aneho|atakpame|ifomess|esgis|ucao|anpe|uk|lbs|université de kara|université de lomé|ul|utt|utbm)\b",
    re.IGNORECASE
)

def detect_entities(question: str):
    entites = {}

    doc = nlp(question)
    for ent in doc.ents:
        entites.setdefault(ent.label_, []).append(ent.text)

    diplome_match = regex_diplomes.findall(question)
    if diplome_match:
        entites.setdefault("diplome", []).extend(diplome_match)

    metier_match = regex_metiers.findall(question)
    if metier_match:
        entites.setdefault("metier", []).extend(metier_match)

    matiere_match = regex_matieres.findall(question)
    if matiere_match:
        entites.setdefault("matiere", []).extend([m[0] for m in matiere_match])

    lieu_match = regex_lieux.findall(question)
    if lieu_match:
        entites.setdefault("lieu", []).extend(lieu_match)

    return entites

# Question : Je viens d'avoir mon bac littéraire mais je ne sais vers quoi me diriger à l'université
def test(questions=None):
    if questions is None:
        questions = ["Je viens d’avoir mon bac série D et je veux savoir les débouchés en médecine au Togo.",
        "Avec une licence en génie logiciel, quels sont les métiers accessibles ?",
        "Quelles matières faut-il pour devenir avocat ?",
        "Je viens d'avoir mon bac littéraire mais je ne sais vers quoi me diriger à l'université"
        ]
        
    result = []
        
    for q in questions:
        result.append(
            {
                "question": q,
                "intention": detect_intention(q),
                "entites": detect_entities(q)
            })
        """
        print(q)
        print(detect_intention(q))
        print(detect_entities(q))
        """
        
    return result
