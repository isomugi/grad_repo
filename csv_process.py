import pandas as pd
import numpy as np
import sys
file = sys.argv[1] 
use_data = ["Temp","Humi","Co2","Ir","Vis","Ill", "Blu", "Sum_Sou","Odr","Sum_Inf","Press"]
#use_data = ["Time","ID","Temp","Humi","Co2","Ir","Full","Vis","Ill", "Blu", "Sum_Sou","Odr","Ult","Sum_Inf","Emotion","Dust","Press"]
df = pd.read_csv(file).replace({0:np.nan, np.inf:np.nan}).dropna(subset = use_data)
count=1
before_time=0
""" def func(x):
    this_time=x["Time"]%10
    if this_time < before_time:
        count += 1
    before_time = this_time
    x["Time"] = this_time + count*1000000 
df = df.apply(func,axis=1)
print(df.head) """
for index,row in df.iterrows():
    df.loc[index,'Time']=row['Time']//10

for index,row in df.iterrows():
    this_time = row['Time']
    if this_time<before_time:
        count+=1
    """ if (this_time+count*1000000) in df.values:
        count+=1 """
    before_time=this_time
    df.loc[index,'Time']=this_time+count*1000000


print(df.head)

