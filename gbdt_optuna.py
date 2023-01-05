import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
use_data = ["Temp","Humi","Co2","Ir","Full","Vis","Ill", "Blu", "Sum_Sou","Odr","Ult","Press"]
file = sys.argv[1]
SEED=42
preprocess = preprocessing.minmax_scale
df = pd.read_csv(file).replace({0:np.nan, np.inf:np.nan}).dropna(subset = use_data)
lenn = len(df)
X = df.loc[:, use_data].astype('float32')
X = preprocess(X)
y = df['Emotion']-1
(X_train, X_test ,y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state = SEED)

train_set=lgb.Dataset(X_train,y_train,free_raw_data=False)
test_set=lgb.Dataset(X_test,y_test,free_raw_data=False)
def objective(trial):
    params = {
        'objective': 'multiclass',    # 多クラス分類を指定
        'metric': 'multi_logloss',    # 損失
        'seed': SEED,
        'num_class':4,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30)
    }
    gbm=lgb.train(params,train_set,valid_sets=test_set,num_boost_round=10000,early_stopping_rounds=100)
    pred_labels=gbm.predict(X_test)
    pred_labels=pred_labels.round(0)
    accuracy=accuracy_score(y_test,pred_labels.argmax(axis=1))
    return accuracy

study=optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=500)
bestparams=study.best_trial.params
bestparams["objective"]="multiclass"
bestparams["metric"]="multi_logloss"
bestparams["seed"]=SEED
bestparams["num_class"]=4
print(bestparams)
model = lgb.train(bestparams,train_set,valid_sets=test_set,num_boost_round=10000,early_stopping_rounds=100)
pred = model.predict(X_test)
pred_labels=pred.round(0)
accuracy=accuracy_score(y_test,pred_labels.argmax(axis=1))
print(accuracy)