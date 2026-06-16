# ML for Finance

Most machine learning models built on financial data look strong in a notebook 
and fail immediately in practice. This repository documents a ground-up approach 
to doing it correctly — addressing stationarity, data leakage, validation 
methodology, and overfitting before a single predictive model is built.

## Notebooks

### financial_ml_fundamentals.ipynb
A rigorous foundation for financial ML — four failure modes identified, 
demonstrated empirically, and resolved.

| Section | Concept | Method |
|--------|---------|--------|
| A1 | Stationarity | ADF test on Nifty 50 prices vs returns |
| A2 | Look-ahead bias | Shuffled vs time-ordered split comparison |
| A3 | Walk-forward validation | Expanding window logistic regression |
| A4 | Overfitting | Unconstrained vs depth-limited decision tree |

**Dataset:** Nifty 50 (^NSEI) via yfinance, 2014–2024  
**Stack:** Python, pandas, scikit-learn, statsmodels