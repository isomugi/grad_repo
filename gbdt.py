import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
import time
import warnings
from sklearn.metrics import accuracy_score
import time

file = sys.argv[1]
use_data = ["Temp","Humi","Co2","Ir","Full","Vis","Ill", "Blu", "Sum_Sou","Odr","Ult","Press"]

preprocess = preprocessing.minmax_scale
seed = 42

#各設定
other_args = sys.argv[2:]
for arg in other_args:
    param, value = arg.split('=')

    if param in ["random_state", "seed"]:
        seed = int(value)

    if param in ["preprocess", "preprocessing", "pre"]:
        if value in ["standard", "std"]:
            preprocess = lambda x: (x - x.mean()) / x.std()

        if value in ["None", "none"]:
            preprocess = lambda x: x

df = pd.read_csv(file).replace({0:np.nan, np.inf:np.nan}).dropna(subset = use_data)
lenn = len(df)

x = df.loc[:, use_data].astype('float32')
x = preprocess(x)
y = df['Emotion']-1 # 正解クラス

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = seed)

model = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=20, random_state=seed)

start_time = time.time()
model.fit(X_train,y_train)

print("fitting time: {}".format(time.time() - start_time))

emotion1 = 0
emotion2 = 0
emotion3 = 0
emotion4 = 0
for t in y_train:
    if t == 1:
        emotion1 += 1
    elif t == 2:
        emotion2 += 1
    elif t == 3:
        emotion3 += 1
    elif t == 4:
        emotion4 += 1     
r1 = float(emotion1)  / (float(emotion1 + emotion2 + emotion3 + emotion4))
r2 = float(emotion2)  / (float(emotion1 + emotion2 + emotion3 + emotion4))
r3 = float(emotion3)  / (float(emotion1 + emotion2 + emotion3 + emotion4))
r4 = float(emotion4)  / (float(emotion1 + emotion2 + emotion3 + emotion4))

print("The number of Data & %d \\" %lenn,end='')
print("\\",end='')
print(" \hline")
print("HappyRatio & " + str(round(r1,3)) + " \\",end='')
print("\\")
print("StressRatio & " + str(round(r2,3)) + " \\",end='')
print("\\")
print("RelaxedRatio & " + str(round(r3,3)) + " \\",end='')
print("\\")
print("SadRatio & " + str(round(r4,3)) + " \\",end='')
print("\\",end='')
print(" \hline")

y_pred=model.predict(X_train)
print('Train Accuracy:', round(accuracy_score(y_train, y_pred), 4))

start_time=time.time()
y_pred=model.predict(X_test)
print('Test Accuracy:', round(accuracy_score(y_test, y_pred), 4))

importance=pd.DataFrame(model.feature_importances_,index=use_data, columns=['importance'])
print(importance)


""" train_set=lgb.Dataset(X_train,y_train,free_raw_data=False)
test_set=lgb.Dataset(X_test,y_test,free_raw_data=False)
params={'lambda_l1': 0.0006023195297354026, 
        'lambda_l2': 0.00013028590695976486, 
        'num_leaves': 165, 
        'feature_fraction': 0.7405420125082255, 
        'bagging_fraction': 0.8850800358233423, 
        'bagging_freq': 5, 
        'min_child_samples': 5, 
        'max_depth': 25, 
        'objective': 'multiclass', 
        'metric': 'multi_logloss', 
        'seed': 42, 
        'num_class': 4}
model = lgb.train(params,train_set,valid_sets=test_set,num_boost_round=10000,early_stopping_rounds=100)
pred = model.predict(X_test)
pred_labels=pred.round(0)
accuracy=accuracy_score(y_test,pred_labels.argmax(axis=1))
print(accuracy) """
