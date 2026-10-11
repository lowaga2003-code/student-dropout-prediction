/* Simple student form: 13 inputs that matter most to our models.
   The other 23 dataset columns stay at typical training values (or a real test student's values). */
var SForm = (function () {
  var S, TEST, BASE, state = {}, ctl = {}, onChange, meta = {};
  var CU = function (s, k) { return "Curricular units " + s + " sem (" + k + ")"; };
  var MAXE = { "1st": 26, "2nd": 23 }, MAXEV = { "1st": 45, "2nd": 33 };
  var VISIBLE = [];
  ["1st", "2nd"].forEach(function (s) { ["enrolled", "evaluations", "approved", "grade"].forEach(function (k) { VISIBLE.push(CU(s, k)); }); });
  VISIBLE.push("Course", "Age at enrollment", "Gender", "Tuition fees up to date", "Debtor");
  var LABEL = { "Marital Status": "Marital status", "Application mode": "Application mode", "Application order": "Application order",
    "Daytime/evening attendance": "Attendance", "Previous qualification": "Previous qualification", "Previous qualification (grade)": "Previous qualification grade",
    "Nacionality": "Nationality", "Mother's qualification": "Mother's qualification", "Father's qualification": "Father's qualification",
    "Mother's occupation": "Mother's occupation", "Father's occupation": "Father's occupation", "Admission grade": "Admission grade",
    "Displaced": "Displaced", "Educational special needs": "Special educational needs", "Scholarship holder": "Scholarship holder",
    "International": "International", "Unemployment rate": "Unemployment rate (%)", "Inflation rate": "Inflation rate (%)", "GDP": "GDP growth (%)" };

  function h(tag, attrs) {
    var el = document.createElement(tag), k, v, i;
    for (k in attrs || {}) {
      v = attrs[k];
      if (v == null || v === false) continue;
      if (k === "text") el.textContent = v;
      else if (k === "class") el.className = v;
      else if (k.slice(0, 2) === "on") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (i = 2; i < arguments.length; i++) [].concat(arguments[i]).forEach(function (c) {
      if (c != null) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return el;
  }
  function show(v, dec) { return dec ? String(Math.round(v * 100) / 100) : String(v); }

  // allowed range of each number input, following the rules seen in the data
  function limits(key) {
    var s = key.indexOf("1st") > 0 ? "1st" : "2nd";
    if (key === "Age at enrollment") return [17, 70];
    if (key === CU(s, "enrolled")) return [0, MAXE[s]];
    if (key === CU(s, "approved")) return [0, state[CU(s, "enrolled")]];
    if (key === CU(s, "evaluations")) return state[CU(s, "enrolled")] === 0 ? [0, 0] : [state[CU(s, "approved")], MAXEV[s]];
    if (key === CU(s, "grade")) return state[CU(s, "approved")] > 0 ? [9.5, 20] : [0, 0];
  }
  function rules() {
    ["1st", "2nd"].forEach(function (s) {
      var e = CU(s, "enrolled"), a = CU(s, "approved"), ev = CU(s, "evaluations"), g = CU(s, "grade");
      if (state[a] > state[e]) state[a] = state[e];                 // cannot pass more units than enrolled
      if (state[e] === 0) state[ev] = 0;                            // no units, no evaluations
      if (state[ev] < state[a]) state[ev] = state[a];               // every approved unit was evaluated
      if (state[a] === 0) state[g] = 0;                             // grade is 0 when nothing is approved
      else if (state[g] === 0) state[g] = 12;
    });
    state["Daytime/evening attendance"] = state.Course === 8014 || state.Course === 9991 ? 0 : 1;   // evening courses
  }

  function stepper(key, label, step, dec) {
    var input = h("input", { id: "f-" + key.replace(/[^A-Za-z0-9]+/g, "-").toLowerCase(), type: "number", step: dec ? "any" : step, "aria-label": label });
    var minus = h("button", { type: "button", tabindex: "-1", "aria-label": "Decrease " + label, text: "−" });
    var plus = h("button", { type: "button", tabindex: "-1", "aria-label": "Increase " + label, text: "+" });
    function commit(v) { var L = limits(key); if (isNaN(v)) { sync(); return; } set(key, Math.min(L[1], Math.max(L[0], v))); }
    minus.addEventListener("click", function () { commit(Math.round((state[key] - step) * 100) / 100); });
    plus.addEventListener("click", function () { commit(Math.round((state[key] + step) * 100) / 100); });
    input.addEventListener("change", function () { if (input.value !== show(state[key], dec)) commit(parseFloat(input.value)); });
    ctl[key] = { kind: "step", input: input, minus: minus, plus: plus, dec: dec };
    return h("span", { class: "step" }, minus, input, plus);
  }
  function seg(key, opts, label) {
    var box = h("div", { class: "seg", role: "group", "aria-label": label });
    var btns = opts.map(function (o) {
      var b = h("button", { type: "button", text: o[1], onclick: function () { set(key, o[0]); } });
      b._v = o[0]; box.appendChild(b); return b;
    });
    ctl[key] = { kind: "seg", btns: btns };
    return box;
  }
  function course() {
    var sel = h("select", { id: "f-course", "aria-label": "Course" });
    S.options.Course.forEach(function (o) { sel.appendChild(h("option", { value: o[0], text: o[1] })); });
    sel.addEventListener("change", function () { set("Course", Number(sel.value)); });
    ctl.Course = { kind: "select", el: sel };
    return sel;
  }

  var presetBtns = [], actualEl, fixedList, fixedTitle;
  function build(form) {
    var PRE = [["graduate", "Typical graduate"], ["enrolled", "Typical enrolled"], ["dropout", "Typical dropout"]];
    var pre = h("div", { class: "presets" });
    PRE.forEach(function (p) {
      var b = h("button", { type: "button", class: "chip", text: p[1], onclick: function () { loadPreset(p[0]); } });
      b._name = p[0]; presetBtns.push(b); pre.appendChild(b);
    });
    var rnd = h("button", { type: "button", class: "chip", text: "Random test student", onclick: function () {
      var i; do { i = Math.floor(Math.random() * TEST.rows.length); } while (meta.source === "test" && i === meta.index); loadTest(i);
    } });
    presetBtns.push(rnd); pre.appendChild(rnd);
    actualEl = h("p", { class: "actual", "aria-live": "polite" });
    form.appendChild(pre); form.appendChild(actualEl);

    var sem = h("div", { class: "sem" }, h("span"), h("span", { class: "colh", text: "1st sem" }), h("span", { class: "colh", text: "2nd sem" }));
    [["enrolled", "Units enrolled", 1, false], ["evaluations", "Evaluations taken", 1, false], ["approved", "Units approved", 1, false], ["grade", "Average grade (/20)", 0.5, true]]
      .forEach(function (r) {
        sem.appendChild(h("span", { class: "rl", text: r[1] }));
        ["1st", "2nd"].forEach(function (s) { sem.appendChild(stepper(CU(s, r[0]), s + " semester " + r[1].toLowerCase(), r[2], r[3])); });
      });
    form.appendChild(h("fieldset", null, h("legend", { text: "Semester results" }), sem));
    form.appendChild(h("fieldset", null, h("legend", { text: "Student" }),
      h("div", { class: "row full" }, h("label", { for: "f-course", text: "Course" }), course()),
      h("div", { class: "row" }, h("span", { text: "Age at enrolment" }), stepper("Age at enrollment", "Age at enrolment", 1, false)),
      h("div", { class: "row" }, h("span", { text: "Gender" }), seg("Gender", [[0, "Female"], [1, "Male"]], "Gender"))));
    form.appendChild(h("fieldset", null, h("legend", { text: "Fees" }),
      h("div", { class: "row" }, h("span", { text: "Tuition fees up to date" }), seg("Tuition fees up to date", [[1, "Yes"], [0, "No"]], "Tuition fees up to date")),
      h("div", { class: "row" }, h("span", { text: "Debtor (owes money)" }), seg("Debtor", [[1, "Yes"], [0, "No"]], "Debtor"))));
    fixedTitle = h("summary");
    fixedList = h("ul");
    form.appendChild(h("details", { class: "fixed" }, fixedTitle, fixedList));
  }

  function nameOf(key, v) {
    var opts = S.options[key];
    if (opts) { for (var i = 0; i < opts.length; i++) if (opts[i][0] === v) return opts[i][1]; return String(v); }
    if (key === "Daytime/evening attendance") return v ? "Daytime" : "Evening";
    if (["Displaced", "Educational special needs", "Scholarship holder", "International"].indexOf(key) >= 0) return v ? "Yes" : "No";
    return String(v);
  }
  function sync() {
    Object.keys(ctl).forEach(function (key) {
      var c = ctl[key], v = state[key];
      if (c.kind === "step") {
        var L = limits(key); c.input.value = show(v, c.dec);
        c.minus.disabled = v <= L[0]; c.plus.disabled = v >= L[1]; c.input.disabled = L[0] === L[1];
      } else if (c.kind === "seg") c.btns.forEach(function (b) { b.setAttribute("aria-pressed", String(b._v === v)); });
      else c.el.value = String(v);
    });
    presetBtns.forEach(function (b) { b.setAttribute("aria-pressed", String(!meta.edited && (meta.source === "preset" ? meta.name === b._name : meta.source === "test" && !b._name))); });
    fixedTitle.textContent = (meta.source === "test" ? "Other 23 details come from test student " + (meta.index + 1) : "Other 23 details are fixed at typical values") + " (show)";
    fixedList.textContent = "";
    Object.keys(LABEL).forEach(function (k) { fixedList.appendChild(h("li", null, LABEL[k] + ": ", h("b", { text: nameOf(k, state[k]) }))); });
  }
  function set(key, v) { state[key] = v; meta.edited = true; rules(); sync(); onChange(JSON.parse(JSON.stringify(state)), meta); }
  function load(rec, m) { state = JSON.parse(JSON.stringify(rec)); meta = m; meta.edited = false; rules(); sync(); onChange(JSON.parse(JSON.stringify(state)), meta); }
  function loadPreset(name) {
    var rec = JSON.parse(JSON.stringify(BASE));
    VISIBLE.forEach(function (k) { rec[k] = S.presets[name][k]; });
    load(rec, { source: "preset", name: name });
  }
  function loadTest(i) {
    var rec = {}; TEST.columns.forEach(function (c, j) { rec[c] = TEST.rows[i][j]; });
    load(rec, { source: "test", index: i, actual: TEST.y[i] });
  }
  function init(o) {
    S = o.schema; TEST = o.test; BASE = S.presets.average; onChange = o.onChange;
    build(o.form); loadPreset(o.start || "graduate");
  }
  return { init: init, h: h, actualEl: function () { return actualEl; } };
})();
