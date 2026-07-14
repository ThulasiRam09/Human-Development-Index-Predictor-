# Human Development Index (HDI) Predictor

A complete ML + Flask project matching your internship spec: predicts a country's
HDI score from Life Expectancy, Mean Years of Schooling, GNI per Capita, and
Internet Users, and classifies it into Low / Medium / High / Very High development.

## Folder structure
```
HDI-Predictor-Project/
├── Dataset/
│   ├── HDI.csv                     144-country dataset (Country, Life expectancy,
│   │                                Mean years of schooling, GNI per capita,
│   │                                Internet users, HDI, HDI Rank)
│   ├── eda_schooling_vs_hdi.png     strip plot (Epic 3)
│   ├── eda_correlation_heatmap.png  correlation heatmap (Epic 3)
│   └── actual_vs_predicted.png      model evaluation scatter plot (Epic 6)
├── Training/
│   ├── generate_dataset.py          builds Dataset/HDI.csv
│   └── train_model.py               EDA -> preprocessing -> train/test split ->
│                                     Linear Regression -> saves Flask/HDI.pkl
└── Flask/
    ├── app.py                       Flask backend (routes: /, /Prediction, /predict)
    ├── HDI.pkl                      trained model (pickle)
    ├── country_encoder.pkl          LabelEncoder for the Country feature
    └── templates/
        ├── home.html                landing page
        ├── indexnew.html            prediction input form
        └── resultnew.html           prediction result page
```

## Note on the dataset
The dataset link referenced in your project PDF
(`github.com/Guided-Projects/HumanDevelopmentIndex`) is no longer live (404).
`Training/generate_dataset.py` builds a 144-country dataset instead, using
realistic country-level indicator values and computing HDI with the official
UNDP formula (geometric mean of health/education/income sub-indices, plus a
touch of noise). This keeps the data internally consistent so the model trains
well (R² ≈ 0.99 on the held-out test set) while covering the whole spectrum
from Norway/Switzerland down to Niger/South Sudan. Swap in a real Kaggle/UNDP
CSV any time — just keep the same column names and rerun `train_model.py`.

## How to run it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn flask

# 1. (optional) regenerate the dataset
cd Training
python generate_dataset.py

# 2. train the model (writes Flask/HDI.pkl and Flask/country_encoder.pkl)
python train_model.py

# 3. run the web app
cd ../Flask
python app.py
```

Then open **http://127.0.0.1:5000** in your browser:
- `/` – Home page introducing the HDI
- `/Prediction` – form to pick a country and enter indicator values
- `/predict` – (POST) returns the predicted HDI score + development tier

## Model performance (on the generated dataset)
- R² ≈ 0.99
- MAE ≈ 0.013
- RMSE ≈ 0.014

## Extending it
- Swap in the real Kaggle HDI dataset (same column names) for production-grade accuracy.
- Add Expected Years of Schooling as a feature to match the full UNDP formula.
- Deploy with `gunicorn`/`waitress` instead of the Flask dev server for production.

# Human Development Index (HDI) Predictor

A complete ML + Flask project matching your internship spec: predicts a country's
HDI score from Life Expectancy, Mean Years of Schooling, GNI per Capita, and
Internet Users, and classifies it into Low / Medium / High / Very High development.

🚀 **Live Demo:** [View the Web App on Render](https://human-development-index-predictor-1-pmwp.onrender.com/)
🎥 **Project Video:** [Watch the Demo Video](https://drive.google.com/file/d/1HQN9cR-cL0Uyo3KtopHqFcQ5Eu_44ehW/view?usp=sharing)

## Folder structure
