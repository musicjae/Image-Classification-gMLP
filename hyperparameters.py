import argparse
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')

parser = argparse.ArgumentParser(description='hyperparameters.....')
parser.add_argument('-b','--batch_size',type=int,default=8)
parser.add_argument('-is','--img_size',type=int,default=256)
parser.add_argument('-e','--epochs',type=int,default=200)
parser.add_argument('-lr','--lr',type=float,default=0.0004)
parser.add_argument('--eps',type=float,default=1e-8)
parser.add_argument('--total_steps',type=int,default=1055*200) # where 200 is epochs
parser.add_argument('--num_classes',type=int,default=4) # final output size
parser.add_argument('--mode',type=str,default='gmlp')

args = parser.parse_args()



