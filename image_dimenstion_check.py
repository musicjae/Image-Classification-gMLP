import os
from PIL import Image
import numpy as np
from data import train_loader

# imgs = os.listdir(os.getcwd()+'/sample/dogs')
#
# for i in imgs:
#     img = Image.open(f'sample/dogs/{i}')
#     if np.array(img).ndim == 3:
#         print(np.array(img).ndim)
#         print(np.array(img).shape)
#         print(type(np.array(img)))
#     else:
#         print(i)

for i,j in enumerate(train_loader.dataset):
    print(j['image_name'])
    print(j['image'].shape)