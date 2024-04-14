import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


def evaluate(model, X_valid: pd.DataFrame, y_valid: pd.Series):
    model_valid_pred = model.predict(X_valid.iloc[:, 2:])
    model_valid_df = X_valid.copy()
    model_valid_df['y'] = y_valid
    model_valid_df['y_pred'] = model_valid_pred
    model_rankic = get_rankic(model_valid_df)
    mse = mean_squared_error(y_valid, model_valid_pred)
    print("mse on validation:", mse)
    return model_rankic, model_valid_df


def perform_pred(model, X_test: pd.DataFrame):
    model_test_pred = model.predict(X_test.iloc[:, 2:])
    model_test_df = X_test.copy()
    model_test_df['y_pred'] = model_test_pred
    return model_test_df
    

def get_rankic(df: pd.DataFrame):
    rankic_values = []
    dates = []
    for date, group in df.groupby('date'):
        pred_rank = group['y_pred'].rank()
        true_rank = group['y'].rank()
        rho, _ = spearmanr(pred_rank, true_rank)
        dates.append(date)
        rankic_values.append(rho)
    mean_rankic = np.mean(rankic_values)
    print('rankic均值为：', mean_rankic)
    return pd.DataFrame({'date': dates, 'RankIC': rankic_values})
