# %%

import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\\Lenovo\\Downloads\\archive (6)\\Summary of Weather.csv')

df.head()

# %%
df2=pd.read_csv('C:\\Users\\Lenovo\\Downloads\\archive (6)\\Weather Station Locations.csv')
df2.head()
# %%
df.isnull().sum().sort_values(ascending=False)

# %%
merged_df=df.merge(df2,left_on='STA',right_on='WBAN',how='left')
# %%
merged_df.isnull().sum().sort_values(ascending=False)
# %%
df.info()
# %%
for col in merged_df.columns:
    if merged_df[col].isnull().sum() == len(merged_df):
        merged_df = merged_df.drop(columns=[col])
# %%
merged_df.isnull().sum().sort_values(ascending=False)
# %%
numeric_cols=[]
numeric_cols = merged_df.select_dtypes(include=['number']).columns

# %%
numeric_cols

# %%
total_rows=len(merged_df)
for col in numeric_cols:
    missing_ratio=merged_df[col].isnull().sum() / total_rows
    if missing_ratio == 0:
        continue
    elif missing_ratio < 0.05:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    elif missing_ratio < 0.4:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    else:
        # Çok eksikse ya drop et ya da özel bir doldurma mantığı uygula
        merged_df = merged_df.drop(columns=[col])
# %%
merged_df.isnull().sum().sort_values(ascending=False)
# %%
merged_df['PRCP'].unique()

# %%
merged_df['PRCP'] = pd.to_numeric(merged_df['PRCP'], errors='coerce')

# %%
merged_df.info()

# %%
merged_df['Snowfall'].unique()
# %%
merged_df['Snowfall'] = pd.to_numeric(merged_df['Snowfall'], errors='coerce')

# %%
merged_df['SNF'].unique()
# %%
merged_df['SNF'] = merged_df['SNF'].replace('T', 0) #olculemeyecek kadar kucuk anlamına gelir
merged_df['SNF'] = pd.to_numeric(merged_df['SNF'], errors='coerce')

# %%
merged_df.isnull().sum().sort_values(ascending=False)
# %%

cols=df[['SNF','Snowfall','PRCP']]
for i in cols:
      merged_df[i] = merged_df[i].fillna(merged_df[i].median())
# %%
merged_df.isnull().sum().sort_values(ascending=False)
# %%
merged_df['PoorWeather'].unique()
# %%
merged_df.drop(columns=['PoorWeather'],inplace=True)
# %%
merged_df.columns
# %%
# TSHDSBRSGF sütunundaki eksikleri 'No Event' ile doldur
merged_df['TSHDSBRSGF'] = merged_df['TSHDSBRSGF'].fillna('No Event')

# Hangi kodları arayacağımızı tanımla
event_codes = ['T', 'S', 'H', 'D', 'SB', 'R', 'SG', 'F']

# Her kod için binary sütun oluştur
for code in event_codes:
    merged_df[f'event_{code}'] = merged_df['TSHDSBRSGF'].str.contains(code, na=False).astype(int)

# Artık orijinal TSHDSBRSGF sütununu silebiliriz
merged_df.drop(columns=['TSHDSBRSGF'], inplace=True)


# %%
merged_df.isnull().sum().sort_values(ascending=False)
# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

# %%
X=merged_df.drop(columns=['MaxTemp'])
y=merged_df['MaxTemp']
# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=15)

# %%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train.dtypes[X_train.dtypes != 'float64'][X_train.dtypes != 'int64']
# %%
# String içeren sütunları bul
problem_cols = X_train.columns[X_train.applymap(lambda x: isinstance(x, str)).any()]
print(problem_cols)

# Bu sütunlardaki örnek değerleri gör
for col in problem_cols:
    print(col, X_train[col].unique()[:20])


X# LAT ve LON'u çıkar
X_train = X_train.drop(columns=['LAT', 'LON'])
X_test = X_test.drop(columns=['LAT', 'LON'])

# Precip'teki 'T' değerlerini 0 yap ve float'a çevir
X_train['Precip'] = X_train['Precip'].replace('T', 0).astype(float)
X_test['Precip'] = X_test['Precip'].replace('T', 0).astype(float)
# Tarih sütununu tamamen çıkar
X_train = X_train.drop(columns=['Date'])
X_test = X_test.drop(columns=['Date'])
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
leak_cols = ['MeanTemp', 'MAX', 'MEA', 'MinTemp', 'MIN']
X_train = X_train.drop(columns=leak_cols, errors='ignore')
X_test = X_test.drop(columns=leak_cols, errors='ignore')


# %%
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# 5. Performans
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
#

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# 2️⃣ Random Forest Model
rf = RandomForestRegressor(
    n_estimators=200,      
    max_depth=None,       
    random_state=42,
    n_jobs=-1            
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 3️⃣ Performans
print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))

# %%
