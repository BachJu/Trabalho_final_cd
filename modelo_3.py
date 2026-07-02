import pre_processamento

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error

import lightgbm as lgb

df_test = pd.read_csv('data/test.csv')

X_train, y_train, X_test = pre_processamento.pre_processamento()

CATEGORICAL_FEATURES = [
    'family', 'store_type', 'holiday_type', 'locale', 'locale_name', 'description', 'city', 'state'
]

# pre_processamento() já dropou a coluna date, mas manteve year/month/day -> reconstituímos
# a data pra fazer um split temporal (últimas 2 semanas do treino viram validação)
datas_train = pd.to_datetime(X_train[['year', 'month', 'day']])
data_corte = datas_train.max() - pd.Timedelta(days=14)

mask_val = datas_train >= data_corte
mask_treino = ~mask_val

X_tr, y_tr = X_train[mask_treino], y_train[mask_treino]
X_val, y_val = X_train[mask_val], y_train[mask_val]

# Métrica oficial da competição é RMSLE -> treinamos em log1p(sales) e revertemos com expm1
y_tr_log = np.log1p(y_tr)

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42,
)

model.fit(
    X_tr, y_tr_log,
    categorical_feature=CATEGORICAL_FEATURES,
)

# Validação local
val_pred = np.expm1(model.predict(X_val))
val_pred = np.clip(val_pred, 0, None)

rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
print(f'RMSLE (validação, últimas 2 semanas do treino): {rmsle:.5f}')

# Previsão final no test.csv da competição
y_pred = np.expm1(model.predict(X_test))
y_pred = np.clip(y_pred, 0, None)

submission = pd.DataFrame({
    'id': df_test['id'],
    'sales': y_pred
})

submission.to_csv('submission_lgbm.csv', index=False)
