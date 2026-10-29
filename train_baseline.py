import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

CAP = 20  # max clips per speaker in training (set to None to use every clip)

def short(a):
    if a.startswith("España"):
        return a.split("(")[0].strip()
    else:
        return a.split(":")[0]

train = pd.read_csv("data/train_filtered.csv")
dev = pd.read_csv("data/dev_filtered.csv")
test = pd.read_csv("data/test_filtered.csv")

if CAP:
    keep = (train.groupby("client_id").cumcount() < CAP).values
else:
    keep = np.ones(len(train), dtype=bool)

y_tr = train.accents.map(short)[keep]
y_dev = dev.accents.map(short)
y_te = test.accents.map(short)

P_tr = np.load("data/train_praat.npy")[keep]
M_tr = np.load("data/train_mfcc.npy")[keep]

P_dev = np.load("data/dev_praat.npy")
M_dev = np.load("data/dev_mfcc.npy")

P_te = np.load("data/test_praat.npy")
M_te = np.load("data/test_mfcc.npy")

feature_sets = {
    "praat only": (P_tr, P_dev, P_te),
    "mfcc only": (M_tr, M_dev, M_te),
    "praat + mfcc": (np.hstack([P_tr, M_tr]), np.hstack([P_dev, M_dev]), np.hstack([P_te, M_te])),
}

print(f"training clips: {len(y_tr)} (cap per speaker: {CAP})")
for name, (X_tr, X_dev, X_te) in feature_sets.items():
    clf = make_pipeline(SimpleImputer(), StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=1000))
    clf.fit(X_tr, y_tr)
    pred_dev = clf.predict(X_dev)
    pred_te = clf.predict(X_te)
    print("=====", name, f"({X_tr.shape[1]} features)")
    print("train accuracy", clf.score(X_tr, y_tr))
    print("dev accuracy", accuracy_score(y_dev, pred_dev), "| macro-F1", f1_score(y_dev, pred_dev, average="macro"))
    print("test accuracy", accuracy_score(y_te, pred_te), "| macro-F1", f1_score(y_te, pred_te, average="macro"))
    print(classification_report(y_te, pred_te, zero_division=0))
    print(f"{'clip':<30}{'true':<30}{'predicted':<30}")
    print("-" * 90)
    for i in range(0, 210, 70):
        print(f"{test.path[i]:<30}{y_te[i]:<30}{pred_te[i]:<30}")