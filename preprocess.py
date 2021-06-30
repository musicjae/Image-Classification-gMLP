import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() # convert string into integer


imgs = os.listdir(os.getcwd()+'/sample/img')
df = pd.DataFrame(imgs)

label_data = np.array(['dog','sink','ref','toilet','two man'])
encoder.fit(label_data)
encoded_label = encoder.transform(label_data)
label = pd.DataFrame(encoded_label)
res = pd.concat([df,label],axis=1)
res.columns = ['images','labels']
res.to_csv('sample/labels.csv')