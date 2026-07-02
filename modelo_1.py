import pre_processamento

import pandas as pd
from sklearn.linear_model import LinearRegression

df_test = pd.read_csv('data/test.csv')

x_train, y_train, x_test = pre_processamento.pre_processamento()

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


submission = pd.DataFrame({
    'id': df_test['id'],
    'sales': y_pred
})

submission.to_csv('submission.csv', index=False)