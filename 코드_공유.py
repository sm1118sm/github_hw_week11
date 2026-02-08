# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target 

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''


####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(random_state=42)

param_grid_xgb = {
    "max_depth": [3, 5, 7, 9, 15],
    "learning_rate": [0.1, 0.01, 0.001],
    "n_estimators": [50, 100, 200, 300]
}

grid_xgb = GridSearchCV(
    xgb_model,
    param_grid_xgb,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

xgb_pred = best_xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("Best parameters:", grid_xgb.best_params_)
print("Best accuracy:", grid_xgb.best_score_)

plt.figure(figsize=(16, 8))
plt.bar(X.columns, best_xgb.feature_importances_)
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()