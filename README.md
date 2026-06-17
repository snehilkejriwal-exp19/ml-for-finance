# ML for Finance

Most machine learning models built on financial data look strong in a notebook 
and fail immediately in practice. This repository documents a ground-up approach 
to doing it correctly — addressing stationarity, data leakage, validation 
methodology, and overfitting before a single predictive model is built.

## Notebooks

### financial_ml_fundamentals.ipynb
A rigorous foundation for financial ML — four failure modes identified, 
demonstrated empirically, and resolved accompanied with feature engineering and then tested by ML models (Logistic Regression, Random Forest Classifer, XGBoost)

| Section | Concept | Method |
|--------|---------|--------|
| A1 | Stationarity | ADF test on Nifty 50 prices vs returns |
| A2 | Look-ahead bias | Shuffled vs time-ordered split comparison |
| A3 | Walk-forward validation | Expanding window logistic regression |
| A4 | Overfitting | Unconstrained vs depth-limited decision tree |
| B | Feature engineering | Return, volume, technical, and macro features on Nifty 50 data |
| C | Model comparison | Logistic Regression, Random Forest, XGBoost — walk-forward evaluated |

## Key findings - NIFTY 50 
- The baseline accuracy came out to 54.39%
- Logistic Regression was the model with the highest accuracy of 56.6% with an edge of 2.2 points over baseline
- Simpler model (Logistic Regression) beat the more complex models (Random Forest(55.7%) & XGBoost(52.7%)) as they make space for noise too rather than just the signal
- Before using a dataset, we should test its stationarity which gives us if we should use a dataset or not by performing adf test.
- Beware of look-ahead bias,i.e, the model studies future data to give the accuracy in the future which results in overfitting
- Always follow walk forward validation,i.e, use the past data to predict the future

**Dataset:** Nifty 50 (^NSEI) via yfinance, 2014–2024  
**Stack:** Python, pandas, scikit-learn, statsmodels

## Author
Snehil Kejriwal — Economics + B.Tech, BITS Pilani Hyderabad
github.com/snehilkejriwal-exp19