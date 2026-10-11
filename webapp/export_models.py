"""Export the trained models (models/*.joblib, saved by the notebooks) as JSON for the browser demo,
together with the raw test students and Python reference predictions used to test the JavaScript."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Dropout", "Enrolled", "Graduate"]
load = lambda name: joblib.load(os.path.join(ROOT, "models", name + ".joblib"))
save = lambda name, obj: json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), separators=(",", ":"))

# ---- the same raw data and split as notebook 1 (the engineered features are recomputed in JavaScript)
df = pd.read_csv(os.path.join(ROOT, "data/raw/students_dropout_academic_success.csv"))
raw_cols = [c for c in df.columns if c != "target"]
for s in ["1st", "2nd"]:
    enrolled = df[f"Curricular units {s} sem (enrolled)"]
    approved = df[f"Curricular units {s} sem (approved)"]
    df[f"approval_rate_{s}"] = np.where(enrolled > 0, approved / enrolled, 0)
df["grade_change"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
df["failed_units"] = (df["Curricular units 1st sem (enrolled)"] - df["Curricular units 1st sem (approved)"]
                      + df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 2nd sem (approved)"])
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre = load("preprocessor")
Xtr, Xte = pre.transform(X_train), pre.transform(X_test)
saved = pd.read_csv(os.path.join(ROOT, "data/processed/test.csv")).drop(columns="target")
print("max |transform - processed test.csv| =", np.abs(Xte - saved.values).max())
Xte_df = saved                                    # exactly the values the notebooks evaluated

# ---- feature recipe for JavaScript, in the exact column order of the preprocessor
spec = []
for name, trans, cols in pre.transformers_:
    if name == "num":
        spec += [{"k": "num", "col": c, "mean": float(m), "std": float(s)} for c, m, s in zip(cols, trans.mean_, trans.scale_)]
    elif name == "bin":
        spec += [{"k": "bin", "col": c} for c in cols]
    elif name == "cat":
        for col, cats, rare in zip(cols, trans.categories_, trans.infrequent_categories_):
            rare = set() if rare is None else set(rare.tolist())
            kept = [int(c) for c in cats if c not in rare]
            spec += [{"k": "cat", "col": col, "val": v} for v in kept]
            if rare:
                spec.append({"k": "rare", "col": col, "known": kept})
assert len(spec) == Xtr.shape[1]


def py_features(row):                       # Python copy of the JavaScript recipe, to check it
    out = []
    for s in spec:
        v = row[s["col"]]
        out.append((v - s["mean"]) / s["std"] if s["k"] == "num" else v if s["k"] == "bin"
                   else float(v == s["val"]) if s["k"] == "cat" else float(v not in s["known"]))
    return out


check = np.array([py_features(r) for _, r in X_test.iterrows()])
print("max |recipe - preprocessor| =", np.abs(check - Xte).max())
assert np.abs(check - Xte).max() < 1e-9

# ---- models
lr, lrb, rf, svm, knn, mlp, km = (load(n) for n in ["logreg", "logreg_binary", "random_forest", "svm", "knn", "mlp", "kmeans"])
assert list(lr.classes_) == CLASSES and list(svm.classes_) == CLASSES


def f32_down(t):
    """largest float32 <= t, written in its shortest form (JavaScript restores it with Math.fround)"""
    a = np.float32(t)
    if float(a) > t:
        a = np.nextafter(a, np.float32(-np.inf))
    return float(np.format_float_positional(a, unique=True, trim="-"))


trees = []
for est in rf.estimators_:
    t = est.tree_
    leaf = t.children_left == -1
    val = t.value[:, 0, :]
    prob = val / val.sum(axis=1, keepdims=True)
    trees.append({"f": [int(v) if not lf else -1 for v, lf in zip(t.feature, leaf)],
                  "t": [f32_down(v) if not lf else 0 for v, lf in zip(t.threshold, leaf)],
                  "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                  "p": [[round(float(v), 6) for v in row] if lf else 0 for row, lf in zip(prob, leaf)]})

compact = lambda A: [[int(v) if float(v).is_integer() else round(float(v), 12) for v in row] for row in A]
models = {
    "lr3": {"coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()},
    "lrb": {"coef": lrb.coef_[0].tolist(), "intercept": float(lrb.intercept_[0])},
    "svm": {"kernel": svm.kernel, "gamma": float(svm._gamma), "nsv": svm.n_support_.tolist(), "sv": compact(svm.support_vectors_),
            "dual": svm.dual_coef_.tolist(), "intercept": svm.intercept_.tolist()},
    "mlp": {"coefs": [w.tolist() for w in mlp.coefs_], "intercepts": [b.tolist() for b in mlp.intercepts_]},
    "kmeans": {"centers": km.cluster_centers_.tolist(), "sizes": np.bincount(km.labels_).tolist(),
               "crosstab": pd.crosstab(km.labels_, y_train.values).reindex(columns=CLASSES).values.tolist()},
}
save("pipeline", {"raw_columns": raw_cols, "features": spec})
save("models_small", models)
save("forest", {"trees": trees})
save("knn", {"X": compact(knn._fit_X), "y": [CLASSES.index(v) for v in y_train], "k": knn.n_neighbors,
             "metric": knn.metric, "weights": knn.weights})

# ---- Python reference outputs on the 885 test students (for test_engine.js)
svm.decision_function_shape = "ovo"
T = Xte_df
ref = {"lr3_prob": lr.predict_proba(T).tolist(), "lr3_pred": lr.predict(T).tolist(),
       "lrb_prob": lrb.predict_proba(T)[:, 1].tolist(),
       "rf_prob": rf.predict_proba(T).tolist(), "rf_pred": rf.predict(T).tolist(),
       "svm_pred": svm.predict(T).tolist(), "svm_dec": svm.decision_function(T).tolist(),
       "knn_prob": knn.predict_proba(T).tolist(), "knn_pred": knn.predict(T).tolist(),
       "mlp_prob": mlp.predict_proba(T).tolist(), "mlp_pred": mlp.predict(T).tolist(),
       "km_cluster": km.predict(T).tolist()}
save("reference", ref)
save("test_features", Xte.tolist())
save("test_raw", {"columns": raw_cols, "rows": X_test[raw_cols].values.tolist(), "y": list(y_test.values)})
save("train_raw", {"columns": raw_cols, "rows": X_train[raw_cols].values.tolist(), "y": list(y_train.values)})

# ---- numbers shown on the pages (from notebook 3's comparison table and the early-warning model)
res = pd.read_csv(os.path.join(ROOT, "results/model_comparison.csv"))
mask_tr, mask_te = y_train != "Enrolled", y_test != "Enrolled"
yb_te = (y_test[mask_te] == "Dropout").astype(int)
pb = lrb.predict(T[mask_te.values]); prb = lrb.predict_proba(T[mask_te.values])[:, 1]
summary = {"comparison": res.drop(columns="Best settings").round(3).replace({np.nan: None}).to_dict(orient="records"),
           "settings": dict(zip(res["Model"], res["Best settings"])),
           "binary": {"accuracy": round(accuracy_score(yb_te, pb), 3), "f1": round(f1_score(yb_te, pb), 3), "auc": round(roc_auc_score(yb_te, prb), 3),
                      "C": float(lrb.C), "class_weight": str(lrb.class_weight)},
           "lr3": {"C": float(lr.C), "class_weight": str(lr.class_weight)},
           "kmeans": {"silhouette": round(float(silhouette_score(pd.read_csv(os.path.join(ROOT, "data/processed/train.csv")).drop(columns="target"), km.labels_)), 3)}}
save("summary", summary)
for f in sorted(os.listdir(OUT)):
    print(f"{f:22s} {os.path.getsize(os.path.join(OUT, f)) / 1024:8.1f} KB")
