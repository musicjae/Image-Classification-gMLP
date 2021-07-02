import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() # convert string into integer


imgs = os.listdir(os.getcwd()+'/sample/img')
df = pd.DataFrame(imgs)

# label_data = np.array(['dog','sink','ref','toilet','two man'])
# encoder.fit(label_data)
# encoded_label = encoder.transform(label_data)
# label = pd.DataFrame(encoded_label)
# res = pd.concat([df,label],axis=1)
# res.columns = ['images','labels']
# res.to_csv('sample/labels.csv')


imgs = pd.DataFrame(os.listdir(os.getcwd()+'/sample/dog-classification/bishon'))
lab1 = pd.DataFrame(['bishon']*len(imgs))

imgs2 = pd.DataFrame(os.listdir(os.getcwd()+'/sample/dog-classification/maltese'))
lab2 = pd.DataFrame(['maltese']*len(imgs2))

imgs3 = pd.DataFrame(os.listdir(os.getcwd()+'/sample/dog-classification/samoyed'))
lab3 = pd.DataFrame(['samoyed']*len(imgs3))

imgs4 = pd.DataFrame(os.listdir(os.getcwd()+'/sample/dog-classification/spitz'))
lab4 = pd.DataFrame(['spitz']*len(imgs4))

result = pd.concat([imgs,lab1],axis=1)
result2 = pd.concat([imgs2,lab2],axis=1)
result3 = pd.concat([imgs3,lab3],axis=1)
result4 = pd.concat([imgs4,lab4],axis=1)

finals = pd.concat([result,result2,result3,result4],axis=0)
finals.columns = ['images','labels']
encoder.fit(finals['labels'])
finals['labels'] = encoder.transform(finals['labels'])
finals.to_csv('sample/dog-clf-labels.csv')
