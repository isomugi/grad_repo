import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import classification_report

use_data = ["Temp","Humi","Co2","Ir","Full","Vis","Ill", "Blu", "Sum_Sou","Odr","Ult","Press"]
#Dust は使わない方がいい."Dust",
file = sys.argv[1]
SEED=42
df = pd.read_csv(file).replace({0:np.nan, np.inf:np.nan}).dropna(subset = use_data)
lenn = len(df)
X = df.loc[:, use_data].astype('float32')
y = df['Emotion'] # 正解クラス

(X_train, X_test ,y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state = 42)
""" pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', RandomForestClassifier(random_state=None))])
pipeline.fit(X_train, y_train)
#y_pred_p = pipeline.predict_proba(X_test)

param_grid={"model__n_estimators": [i for i in range(10, 25)],
            "model__max_depth":[i for i in range(10, 25)],
            }
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=cv)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_,grid_search.best_score_)
for params, mean_test_score in grid_search.cv_results_:
    print(mean_test_score,params) """
"""     print("\n+ テストデータでの識別結果:\n")
    y_true, y_pred = y_test, grid_search.predict(X_test)
    print(classification_report(y_true, y_pred)) """
""" 
max_score = 0
SearchMethod = 0 """
param_grid={"n_estimators": [i for i in range(15, 25)],
            "max_depth":[i for i in range(15, 25)],
            }

model=RandomForestClassifier(random_state=0)
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=cv)
clf.fit(X, y)
""" cv_result = pd.DataFrame(clf.cv_results_)
cv_result """
print(clf.best_estimator_,clf.best_score_)
pred=clf.best_estimator_.predict(X_test)
score=accuracy_score(y_test,pred)
print(score)
""" for model, param in tqdm(param_grid.items()):
    clf = GridSearchCV(model, param)
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)
    score = f1_score(y_test, pred_y, average="micro")
    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param)) """

""" model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("")
print("デフォルトスコア:", score) """