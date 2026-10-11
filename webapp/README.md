# Web demo

The demo pages in [`docs/`](../docs) run every model **in the browser**. There is no server and no Python at prediction time.

## How it works

1. `export_models.py` loads the models saved by the notebooks (`models/*.joblib`) and exports their parameters as JSON: the preprocessing recipe (scaler means and standard deviations, one-hot categories), the regression weights, the 150 Random Forest trees, the SVM support vectors, the KNN training points, the neural network weights and the K-Means centres. It also saves Python's predictions for the 885 test students.
2. `make_schema.py` writes readable labels for the coded columns and the preset students used by the forms.
3. `engine.js` turns a raw student record into the same 110 features as notebook 1 (including the 4 engineered features) and computes every model's prediction from the exported parameters.
4. `build_pages.py` inlines the CSS, JavaScript and JSON into two self-contained pages: `docs/index.html` and `docs/compare.html`.

`test_engine.js` checks the JavaScript against scikit-learn on all 885 test students. The current build has no prediction differences, and the largest probability difference is below 0.0000001.

## Rebuild

Run the three notebooks first (they save the models), then:

```bash
cd webapp
python export_models.py
python make_schema.py
python build_pages.py
node test_engine.js      # optional check (needs Node.js)
```

## Publish with GitHub Pages

Repository **Settings → Pages → Build and deployment**: Source *Deploy from a branch*, branch `main`, folder `/docs`. The demo appears at `https://lowaga2003-code.github.io/student-dropout-prediction/`.
