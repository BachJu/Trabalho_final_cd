import pandas as pd
from sklearn.preprocessing import LabelEncoder

def pre_processamento():
    # Leitura dos csv
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    df_holidays = pd.read_csv('data/holidays_events.csv')
    df_oil = pd.read_csv('data/oil.csv')
    df_stores = pd.read_csv('data/stores.csv')

    # Normalização da coluna date
    for df in [df_train, df_test, df_holidays, df_oil]:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.day_of_week

    holidays = df_holidays.drop_duplicates(subset=['date'])

    # Juntar os csv em train e test
    def juntar_tabelas(df):
        df = pd.merge(df, df_stores, on='store_nbr', how='left')
        
        df = pd.merge(df, df_oil[['date', 'dcoilwtico']], on='date', how='left')
        df['dcoilwtico'] = df['dcoilwtico'].fillna(df['dcoilwtico'].mean())
        
        df = pd.merge(df, holidays[['date', 'type', 'locale', 'locale_name', 'description']], on='date', how='left')
        df = df.rename(columns={'type_x': 'store_type', 'type_y': 'holiday_type'})
            
        return df

    df_train = juntar_tabelas(df_train)
    df_test = juntar_tabelas(df_test)

    # Normalizção das colunas -> ao usar regressor precisa ser valores numericos
    for coluna in ['family', 'store_type', 'holiday_type', 'locale', 'locale_name', 'description', 'city', 'state']:
        le = LabelEncoder()

        df_train[coluna] = df_train[coluna].astype(str)
        df_test[coluna] = df_test[coluna].astype(str)
        
        df_train[coluna] = le.fit_transform(df_train[coluna])
        df_test[coluna] = le.transform(df_test[coluna])

    # Valores que vamos usar no calculo
    X_train = df_train.drop(columns=['id', 'sales', 'date'])
    X_test = df_test.drop(columns=['id', 'date'])

    y_train = df_train['sales']

    return X_train, y_train, X_test