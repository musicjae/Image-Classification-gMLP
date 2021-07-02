import numpy as np
import tqdm
from hyperparameters import args
from torchvision import transforms
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import os


## To get mean, std
def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    # tqdm은 진행상태를 알려주는 함수
    for _,data in enumerate(dataloader):
        batch_samples = data['image'].size(0)
        data = data['image'].view(batch_samples, data['image'].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(),std.numpy()


mean,std = np.array([0.34903467, 0.31960577, 0.2967486]), np.array([0.25601476, 0.2398773,  0.23781188])
inference_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(args.img_size, args.img_size)),
                    transforms.Normalize(mean,std)])