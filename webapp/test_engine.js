// Check the JavaScript engine against scikit-learn on all 885 raw test students.  Run: node test_engine.js
const fs = require("fs");
const J = (f) => JSON.parse(fs.readFileSync(__dirname + "/data/" + f + ".json"));
const E = require("./engine.js");
const small = J("models_small");
E.init(J("pipeline"), { lr3: small.lr3, lrb: small.lrb, svm: small.svm, mlp: small.mlp, kmeans: small.kmeans,
  forest: J("forest"), knn: J("knn") });

const test = J("test_raw"), ref = J("reference"), Xref = J("test_features"), C = E.CLASSES;
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const worst = { features: 0, lr3: 0, lrb: 0, forest: 0, svm: 0, knn: 0, mlp: 0 }, bad = {};
const miss = (k, i) => (bad[k] = bad[k] || []).push(i);
const t0 = Date.now();
test.rows.forEach((row, i) => {
  const r = {}; test.columns.forEach((c, j) => (r[c] = row[j]));
  const x = E.features(r);
  worst.features = Math.max(worst.features, maxAbs(x, Xref[i]));
  const a = E.lr3(x); worst.lr3 = Math.max(worst.lr3, maxAbs(a.prob, ref.lr3_prob[i])); if (C[a.cls] !== ref.lr3_pred[i]) miss("lr3", i);
  worst.lrb = Math.max(worst.lrb, Math.abs(E.lrb(x) - ref.lrb_prob[i]));
  const f = E.forest(x); worst.forest = Math.max(worst.forest, maxAbs(f.prob, ref.rf_prob[i])); if (C[f.cls] !== ref.rf_pred[i]) miss("forest", i);
  const s = E.svm(x); worst.svm = Math.max(worst.svm, maxAbs(s.dec, ref.svm_dec[i])); if (C[s.cls] !== ref.svm_pred[i]) miss("svm", i);
  const k = E.knn(x); worst.knn = Math.max(worst.knn, maxAbs(k.prob, ref.knn_prob[i])); if (C[k.cls] !== ref.knn_pred[i]) miss("knn", i);
  const m = E.mlp(x); worst.mlp = Math.max(worst.mlp, maxAbs(m.prob, ref.mlp_prob[i])); if (C[m.cls] !== ref.mlp_pred[i]) miss("mlp", i);
  if (E.kmeans(x).cluster !== ref.km_cluster[i]) miss("kmeans", i);
});
console.log("largest differences:", worst);
console.log("prediction mismatches:", Object.keys(bad).length ? bad : "none");
console.log(`${test.rows.length} students x 7 models in ${Date.now() - t0} ms`);
