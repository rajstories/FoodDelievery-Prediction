# Food Delivery Time Prediction

Predict how many **minutes** a food delivery will take, given the order details —
who is delivering it, how far it has to travel, the weather, traffic, time of day,
and more. Framed as a **supervised regression** problem and solved end-to-end, from
raw messy data to a saved model ready for inference.

---

## Results

Four models were trained and compared. XGBoost (with tuned hyperparameters) came out on top.

| Model | Test R² | Test RMSE (min) | Train R² | Notes |
|---|---|---|---|---|
| Linear Regression | 0.57 | 6.21 | 0.58 | Baseline — relationships aren't linear |
| Decision Tree | 0.67 | 5.46 | 1.00 | Overfits (memorises the training set) |
| Random Forest | 0.82 | 4.03 | 0.97 | Big gain — averaging tames overfitting |
| **XGBoost (tuned)** | **0.83** | **3.91** | 0.85 | Best score *and* healthiest train/test gap |

*RMSE is in minutes, so the final model is off by roughly 4 minutes on average.*

---

## Approach

1. **Load & inspect** the raw data
2. **Clean** messy text and missing values (e.g. `"conditions Sunny"` → `Sunny`, `"(min) 24"` → `24`)
3. **EDA** — find which features actually move delivery time
4. **Feature engineering**
   - `distance_km` from restaurant/customer coordinates via the **Haversine formula**
   - `order_hour` extracted from the order timestamp
5. **Encode** categoricals (ordinal for traffic density, one-hot for the rest) and **scale** features
6. **Train & compare** Linear Regression, Decision Tree, Random Forest, XGBoost
7. **Tune** the best model (RandomizedSearchCV), **evaluate** (RMSE, R², Adjusted R²), and **save** it for inference

The maths and reasoning behind each step are explained inline in the notebook.

---

## Dataset

The dataset contains historical food-delivery orders with features such as the
delivery person's age and rating, restaurant and delivery coordinates, weather,
road-traffic density, vehicle type, order type, festival flag, and city type.
The target is `Time_taken(min)`.

---

## Tech stack

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · XGBoost · joblib

---

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/rajstories/FoodDelivery_Prediction.git
cd FoodDelivery_Prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Food_Delivery_Time_Prediction.ipynb
```

The notebook reads the CSV from the repo root, so it runs as-is with no path edits.

---

## Repository structure

```
FoodDelivery_Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── Food_Delivery_Time_Prediction.ipynb   # the full, documented notebook
├── Food delivery.csv                     # dataset (45,593 orders)
└── generated model artifacts are created locally and ignored by git
```

---

## Credits

Based on [Manoj00018/Food-Delivery-Time-Prediction](https://github.com/Manoj00018/Food-Delivery-Time-Prediction),
licensed under Apache 2.0 (see `LICENSE`). This copy adds `requirements.txt`, `.gitignore`,
the trained `.pkl` artefacts, and small fixes so the notebook runs locally instead of on Colab.

---

## Reproducibility

The notebook was re-run end to end on Python 3.14 with the pinned versions in
`requirements.txt`; the `.pkl` files in this repo are the artefacts of that run.
Reproduced scores match the table above (tuned XGBoost: **test R² 0.829, RMSE 3.91 min**).

Tuned XGBoost hyperparameters found by `RandomizedSearchCV`:

```python
{'subsample': 0.9, 'reg_lambda': 2, 'reg_alpha': 0, 'n_estimators': 500,
 'max_depth': 8, 'learning_rate': 0.01, 'colsample_bytree': 1.0}
```
