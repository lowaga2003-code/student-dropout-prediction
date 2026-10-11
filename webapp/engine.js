/* Prediction engine for the web demo.
   Raw student record -> the same 110 features as the notebooks (4 engineered features, standard scaling,
   one-hot encoding with rare categories grouped) -> every trained model.
   Works in the browser and in Node.js (see test_engine.js). */
var Engine = (function () {
  var spec = [], M = {}, CLASSES = ["Dropout", "Enrolled", "Graduate"];

  function engineered(r) {
    var e1 = r["Curricular units 1st sem (enrolled)"], a1 = r["Curricular units 1st sem (approved)"];
    var e2 = r["Curricular units 2nd sem (enrolled)"], a2 = r["Curricular units 2nd sem (approved)"];
    return {
      approval_rate_1st: e1 > 0 ? a1 / e1 : 0,
      approval_rate_2nd: e2 > 0 ? a2 / e2 : 0,
      grade_change: r["Curricular units 2nd sem (grade)"] - r["Curricular units 1st sem (grade)"],
      failed_units: (e1 - a1) + (e2 - a2)
    };
  }

  function init(pipeline, models) {
    spec = pipeline.features; M = models || {};
    if (M.forest) M.forest.trees.forEach(function (t) { t.t = t.t.map(Math.fround); });   // exact float32 thresholds
  }

  // one value per feature, in the preprocessor's column order
  function features(r) {
    var eng = engineered(r), x = new Array(spec.length);
    for (var j = 0; j < spec.length; j++) {
      var s = spec[j], v = s.col in eng ? eng[s.col] : r[s.col];
      if (s.k === "num") x[j] = (v - s.mean) / s.std;                  // StandardScaler
      else if (s.k === "bin") x[j] = v;                                // 0/1 column, unchanged
      else if (s.k === "cat") x[j] = v === s.val ? 1 : 0;              // one-hot column
      else x[j] = s.known.indexOf(v) < 0 ? 1 : 0;                      // "infrequent" column: any rare or unseen category
    }
    return x;
  }

  function dot(w, x) { var s = 0; for (var j = 0; j < x.length; j++) s += w[j] * x[j]; return s; }
  function softmax(z) {
    var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - m); });
    var t = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / t; });
  }
  function argmax(a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

  // 1) Logistic Regression: softmax over 3 classes, and the sigmoid early-warning model (Dropout vs Graduate)
  function lr3(x) {
    var p = softmax(M.lr3.coef.map(function (w, k) { return dot(w, x) + M.lr3.intercept[k]; }));
    return { prob: p, cls: argmax(p) };
  }
  function lrb(x) { return 1 / (1 + Math.exp(-(dot(M.lrb.coef, x) + M.lrb.intercept))); }

  // 2) Random Forest: average of the leaf probabilities of every tree (trees compare float32 values)
  function forest(x) {
    var xf = x.map(Math.fround), T = M.forest.trees, p = [0, 0, 0], votes = [0, 0, 0];
    for (var i = 0; i < T.length; i++) {
      var t = T[i], n = 0;
      while (t.f[n] !== -1) n = xf[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
      var leaf = t.p[n];
      for (var k = 0; k < 3; k++) p[k] += leaf[k];
      votes[argmax(leaf)]++;
    }
    p = p.map(function (v) { return v / T.length; });
    return { prob: p, cls: argmax(p), votes: votes, trees: T.length };
  }

  // 3) SVM (RBF or linear kernel): one-vs-one duels (D,E), (D,G), (E,G), the class with most wins is predicted
  function svm(x) {
    var S = M.svm, n = S.sv.length, K = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var row = S.sv[i], v = 0;
      if (S.kernel === "rbf") { for (var j = 0; j < x.length; j++) { var d = row[j] - x[j]; v += d * d; } K[i] = Math.exp(-S.gamma * v); }
      else { for (var j2 = 0; j2 < x.length; j2++) v += row[j2] * x[j2]; K[i] = v; }
    }
    var start = [0, S.nsv[0], S.nsv[0] + S.nsv[1]], votes = [0, 0, 0], dec = [], pair = 0;
    for (var a = 0; a < 3; a++) for (var b = a + 1; b < 3; b++) {
      var s = S.intercept[pair];
      for (var u = 0; u < S.nsv[a]; u++) s += S.dual[b - 1][start[a] + u] * K[start[a] + u];
      for (var w = 0; w < S.nsv[b]; w++) s += S.dual[a][start[b] + w] * K[start[b] + w];
      dec.push(s); votes[s > 0 ? a : b]++; pair++;
    }
    return { votes: votes, dec: dec, cls: argmax(votes) };
  }

  // 4) KNN: the k nearest training students (Manhattan or Euclidean distance), uniform or distance-weighted vote
  function knn(x) {
    var K = M.knn, X = K.X, n = X.length, d = new Float64Array(n), idx = new Array(n);
    for (var i = 0; i < n; i++) {
      var row = X[i], s = 0;
      if (K.metric === "manhattan") for (var j = 0; j < x.length; j++) s += Math.abs(row[j] - x[j]);
      else { for (var j2 = 0; j2 < x.length; j2++) { var t = row[j2] - x[j2]; s += t * t; } s = Math.sqrt(s); }
      d[i] = s; idx[i] = i;
    }
    idx.sort(function (a, b) { return d[a] - d[b] || a - b; });
    var near = idx.slice(0, K.k), zero = near.some(function (q) { return d[q] === 0; }), score = [0, 0, 0];
    near.forEach(function (q) {
      var wgt = K.weights === "distance" ? (zero ? (d[q] === 0 ? 1 : 0) : 1 / d[q]) : 1;
      score[K.y[q]] += wgt;
    });
    var tot = score[0] + score[1] + score[2], p = score.map(function (v) { return v / tot; });
    return { prob: p, cls: argmax(p), neighbours: near };
  }

  // 5) Neural network (MLP): ReLU hidden layers, softmax output
  function mlp(x) {
    var C = M.mlp.coefs, B = M.mlp.intercepts, a = x;
    for (var l = 0; l < C.length; l++) {
      var out = B[l].slice();
      for (var i = 0; i < a.length; i++) { var v = a[i]; if (v !== 0) for (var o = 0; o < out.length; o++) out[o] += v * C[l][i][o]; }
      a = l < C.length - 1 ? out.map(function (z) { return z > 0 ? z : 0; }) : out;
    }
    var p = softmax(a);
    return { prob: p, cls: argmax(p) };
  }

  // 6) K-Means: the nearest cluster centre (no labels used)
  function kmeans(x) {
    var best = 0, bestD = Infinity;
    M.kmeans.centers.forEach(function (c, k) {
      var s = 0; for (var j = 0; j < x.length; j++) { var t = x[j] - c[j]; s += t * t; }
      if (s < bestD) { bestD = s; best = k; }
    });
    return { cluster: best };
  }

  return { init: init, features: features, engineered: engineered, lr3: lr3, lrb: lrb, forest: forest,
           svm: svm, knn: knn, mlp: mlp, kmeans: kmeans, CLASSES: CLASSES };
})();
if (typeof module !== "undefined") module.exports = Engine;
