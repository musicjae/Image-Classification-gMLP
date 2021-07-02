from model import *
from hyperparameters import args
from pytorch_lightning.callbacks import ModelCheckpoint
from hyperparameters import args
import random
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb

### Setting for Nice Training###
random_seed = 827
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

wandb_logger = WandbLogger()

wandb.init()
wandb.config.update(args)

checkpoint_callback = ModelCheckpoint(
    #monitor=val_loss
    dirpath='lightning_logs/',
    filename='trained-{epoch:02d}-{val_loss:.2f}'+f'-{args.mode}',
    #save_top_k=3,
    #mode='min',
)
###############################

trainer = pl.Trainer(max_epochs=20,progress_bar_refresh_rate=20,
                     auto_scale_batch_size=True,
                     logger=wandb_logger,
                     callbacks=[checkpoint_callback]) # bach size auto finder

tuner = tuner.tuning.Tuner(trainer)

if args.mode == 'gmlp':
    wandb.watch(gmlp)
    print('Traing using gMLP...')
    trainer.fit(gmlp,train_loader)
    trainer.validate(val_dataloaders=validation_loader)
    trainer.test(test_dataloaders=test_loader)


elif args.mode == 'basic-mlp':
    wandb.watch(basic_mlp)
    print('Traing using basic-MLP...')
    trainer.fit(basic_mlp, train_loader)
    trainer.validate(val_dataloaders=validation_loader)
    trainer.test(test_dataloaders=test_loader)