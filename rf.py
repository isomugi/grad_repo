import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import sys
import time

use_data = ["Temp","Humi","Co2","Ir","Vis","Ill", "Blu", "Sum_Sou","Odr","Sum_Inf","Press"]
#"Ult","Full"
file = sys.argv[1]

preprocess = preprocessing.minmax_scale
seed = 42
""" max_depth = 20
n_estimators = 18 """
max_depth = 20
n_estimators = 24
evaluate_method = "holdout"

other_args = sys.argv[2:]
for arg in other_args:
    param, value = arg.split('=')

    if param in ["preprocess", "preprocessing", "pre"]:
        if value in ["standard", "std"]:
            preprocess = lambda x: (x - x.mean()) / x.std()

        if value in ["None", "none"]:
            preprocess = lambda x: x

    if param in ["random_state", "seed"]:
        seed = int(value)

    if param in ["max_depth"]:
        max_depth = int(value)

    if param in ["n_estimators", "num_estimators"]:
        n_estimators = int(value)

    if param in ["evaluate", "evaluate_method"]:
        if value in ["k_value"]:
            evaluate_method = "k_value"

df = pd.read_csv(file).replace({0:np.nan, np.inf:np.nan}).dropna(subset = use_data)
df = df[df["Dust"] > 0]
lenn = len(df)
x = df.loc[:, use_data].astype('float32')
x = preprocess(x)

y = df['Emotion'] # 正解クラス
print(df.isnull().all())

if evaluate_method == "k_value":
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state = seed)
    scores = cross_val_score(model, x, y)

    print('Cross-Validation scores: {}'.format(scores))
    print('Average score: {}'.format(round(np.mean(scores), 3)))
    exit()

(X_train, X_test ,y_train, y_test) = train_test_split(x, y, test_size = 0.3, random_state = seed)

model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state = seed)

start_time = time.time()
model.fit(X_train, y_train)
print("fitting time: {}".format(time.time() - start_time))

start_time = time.time()
y_pred = model.predict(X_test)
print("test time: {}".format(time.time() - start_time))

emotion1 = 0
emotion2 = 0
emotion3 = 0
emotion4 = 0
for e in y_train:
    if e == 1:
        emotion1 += 1
    elif e == 2:
        emotion2 += 1
    elif e == 3:
        emotion3 += 1
    elif e == 4:
        emotion4 += 1     
ratio1 = float(emotion1)  / (float(emotion1 + emotion2 + emotion3 + emotion4))
ratio2 = float(emotion2)  / (float(emotion1 + emotion2 + emotion3 + emotion4))
ratio3 = float(emotion3)  / (float(emotion1 + emotion2 + emotion3 + emotion4))
ratio4 = float(emotion4)  / (float(emotion1 + emotion2 + emotion3 + emotion4))

print("The number of Data & %d \\" %lenn,end='')
print("\\",end='')
print(" \hline")
print("HappyRatio & " + str(round(ratio1,3)) + " \\",end='')
print("\\")
print("StressRatio & " + str(round(ratio2,3)) + " \\",end='')
print("\\")
print("RelaxedRatio & " + str(round(ratio3,3)) + " \\",end='')
print("\\")
print("SadRatio & " + str(round(ratio4,3)) + " \\",end='')
print("\\",end='')
print(" \hline")

#モデルを作成する段階でのモデルの識別精度
trainaccuracy_model = model.score(X_train, y_train)
print('TrainAccuracy & '+ str(round(trainaccuracy_model,4)) + " \\", end='')
print("\\")
#作成したモデルに学習に使用していない評価用のデータセットを入力し精度を確認
accuracy_model = accuracy_score(y_test, y_pred)
print('Accuracy & ' + str(round(accuracy_model,3)) + " \\", end='')
print("\\")

pre_emotion1 = 0
pre_emotion2 = 0
pre_emotion3 = 0
pre_emotion4 = 0
for e in y_pred:
    if e == 1:
        pre_emotion1 += 1
    elif e == 2:
        pre_emotion2 += 1
    elif e == 3:
        pre_emotion3 += 1
    elif e == 4:
        pre_emotion4 += 1 
pratio1 = float(pre_emotion1)  / (float(pre_emotion1 + pre_emotion2 + pre_emotion3 + pre_emotion4))
pratio2 = float(pre_emotion2)  / (float(pre_emotion1 + pre_emotion2 + pre_emotion3 + pre_emotion4))
pratio3 = float(pre_emotion3)  / (float(pre_emotion1 + pre_emotion2 + pre_emotion3 + pre_emotion4))
pratio4 = float(pre_emotion4)  / (float(pre_emotion1 + pre_emotion2 + pre_emotion3 + pre_emotion4))
print(pre_emotion1)
print(pre_emotion2)
print(pre_emotion3)
print(pre_emotion4)
print(pratio1)
print(pratio2)
print(pratio3)
print(pratio4)
print()

importance = dict(zip(use_data, model.feature_importances_))
importance = sorted(importance.items(), key = lambda x: x[1], reverse = True)

for key, value in importance:
    print("{}:\e{}".format(key, round(value, 3)))