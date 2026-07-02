import pre_processamento

import pandas as pd
import numpy as np

import xgboost as xgb

df_test = pd.read_csv('data/test.csv')

X_train, y_train, X_test = pre_processamento.pre_processamento()

# Treinamento do modelo
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)

submission = pd.DataFrame({
    'id': df_test['id'],
    'sales': y_pred
})

submission.to_csv('submission.csv', index=False)