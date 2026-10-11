"""Build the two demo pages into ../docs for GitHub Pages:
index.html (Logistic Regression risk predictor) and compare.html (all six model families).
Run after export_models.py and make_schema.py."""
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
read = lambda p: open(os.path.join(HERE, p), encoding="utf-8").read()
J = lambda f: json.load(open(os.path.join(HERE, "data", f + ".json"), encoding="utf-8"))
META = {
    "index": ("Dropout Risk Predictor", "Will a student drop out, stay enrolled or graduate? A Logistic Regression model that runs in your browser."),
    "compare": ("Six Models, One Student", "Six machine learning models predict the same student side by side, all running in the browser."),
}
REPO_LINK = """<script>(function () { var h = location.hostname, p = location.pathname.split("/")[1];
  if (/\\.github\\.io$/.test(h) && p) { var a = document.getElementById("repoLink");
    a.href = "https://github.com/" + h.split(".")[0] + "/" + p; a.hidden = false; } })();</script>"""


def tag(id_, obj):
    return f'<script type="application/json" id="{id_}">' + json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/") + "</script>"


def build(src, name, tags, fill=None):
    html = read(src)
    for k, v in {"/*__BASE_CSS__*/": read("src/base.css"), "/*__ENGINE__*/": read("engine.js"),
                 "/*__FORM__*/": read("src/form.js"), "<!--__DATA__-->": "\n".join(tags), **(fill or {})}.items():
        assert k in html, (src, k)
        html = html.replace(k, v)
    title, desc = META[name]
    cut = html.index("</style>") + len("</style>")
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'<meta name="description" content="{desc}">\n<meta property="og:title" content="{title}">\n'
            f'<meta property="og:description" content="{desc}">\n<meta property="og:type" content="website">\n')
    full = head + html[:cut] + "\n</head>\n<body>\n" + html[cut:] + "\n" + REPO_LINK + "\n</body>\n</html>\n"
    open(os.path.join(DOCS, name + ".html"), "w", encoding="utf-8").write(full)
    print(f"docs/{name}.html  {len(full.encode()) / 1024:,.0f} KB")


def weight_text(cw):
    if isinstance(cw, dict):
        return f"extra weight {cw['Enrolled']} on Enrolled"
    return "balanced class weights" if cw == "balanced" else "no class weights"


small, pipeline, schema, test, forest, knn, S = (J(n) for n in ["models_small", "pipeline", "schema", "test_raw", "forest", "knn", "summary"])
R = {r["Model"]: r for r in S["comparison"]}
P = {m: ast.literal_eval(s) for m, s in S["settings"].items()}
lr, rf, sv, kn, nn = P["Logistic Regression"], P["Random Forest"], P["SVM"], P["KNN"], P["Neural Network (MLP)"]
rows = [
    ("Logistic Regression", "Logistic Regression", f"C = {lr['C']} · {weight_text(lr['class_weight'])}"),
    ("Random Forest", "Random Forest", f"{len(forest['trees'])} trees · max depth {rf['max_depth']} · min {rf['min_samples_leaf']} per leaf · {weight_text(rf['class_weight'])}"),
    ("SVM", f"SVM ({sv['kernel'].upper() if sv['kernel'] == 'rbf' else sv['kernel']} kernel)", f"C = {sv['C']} · {weight_text(sv['class_weight'])}"),
    ("KNN", f"KNN (k = {kn['n_neighbors']})", f"{kn['metric'].capitalize()} distance · {'distance-weighted' if kn['weights'] == 'distance' else 'uniform'} vote"),
    ("Neural Network (MLP)", "Neural network (MLP)", f"{' → '.join(str(n) for n in nn['hidden_layer_sizes'])} neurons · alpha {nn['alpha']}"),
]
table = [{"name": label, "how": how, "main": key == "Logistic Regression", "cv": R[key]["CV macro F1"], "acc": R[key]["Test accuracy"],
          "f1": R[key]["Test macro F1"], "auc": R[key]["ROC AUC"], "gap": R[key]["Train-test gap"]} for key, label, how in rows]
table.append({"name": "K-Means (k = 3)", "how": "unsupervised · never sees the outcome", "sil": S["kmeans"]["silhouette"]})

# the sentence under the table is checked against the numbers, so a retrained model cannot leave it wrong
top3 = sorted(S["comparison"], key=lambda r: -r["Test macro F1"])[:3]
gaps = {r["Model"]: r["Train-test gap"] for r in S["comparison"]}
assert "Logistic Regression" in [r["Model"] for r in top3] and min(gaps, key=gaps.get) == "Logistic Regression"
spread = top3[0]["Test macro F1"] - top3[2]["Test macro F1"]
names = {"Logistic Regression": "Logistic Regression", "Random Forest": "Random Forest", "Neural Network (MLP)": "the neural network", "SVM": "the SVM", "KNN": "KNN"}
order = [names[r["Model"]] for r in top3]
note = (f"{order[0][0].upper() + order[0][1:]}, {order[1]} and {order[2]} finish within {spread:.3f} macro F1 of each other. "
        f"Logistic Regression is the main model: it has the smallest train–test gap ({gaps['Logistic Regression']:.3f}) and its weights can be explained. "
        "The SVM was trained without probabilities, so it has no ROC-AUC.")

os.makedirs(DOCS, exist_ok=True)
open(os.path.join(DOCS, ".nojekyll"), "w").close()
fill = {"%%LR_ACC%%": f"{R['Logistic Regression']['Test accuracy']:.3f}", "%%LR_F1%%": f"{R['Logistic Regression']['Test macro F1']:.3f}",
        "%%B_ACC%%": f"{S['binary']['accuracy']:.3f}", "%%B_F1%%": f"{S['binary']['f1']:.3f}",
        "%%LR_SET%%": f"C = {lr['C']} with {weight_text(lr['class_weight'])}",
        "%%B_SET%%": f"C = {S['binary']['C']:g} with {weight_text(S['binary']['class_weight'])}"}
build("src/predictor.html", "index", [tag("d-pipeline", pipeline), tag("d-models", {"lr3": small["lr3"], "lrb": small["lrb"]}),
      tag("d-schema", schema), tag("d-test", test)], fill)
build("src/compare.html", "compare", [tag("d-pipeline", pipeline), tag("d-models", small), tag("d-schema", schema), tag("d-test", test),
      tag("d-forest", forest), tag("d-knn", knn), tag("d-table", {"rows": table, "note": note})])
print("table note:", note)
