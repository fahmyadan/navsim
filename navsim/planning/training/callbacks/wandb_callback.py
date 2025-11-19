import  pytorch_lightning as pl 
import wandb 
from omegaconf import OmegaConf
from copy import deepcopy

class LogConfigCallback(pl.Callback):
    def __init__(self, cfg):
        self.cfg = dict(deepcopy(cfg))
        del self.cfg['train_logs']
        del self.cfg['val_logs']
        del self.cfg['test_logs']
        self.cfg = OmegaConf.create(self.cfg)

    def on_fit_start(self, trainer, pl_module):
        logger = trainer.logger.experiment 
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        logger.config.update(cfg_dict, allow_val_change=True)