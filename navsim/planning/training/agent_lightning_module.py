from typing import Any, Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.metrics.compute_metrics import get_scores
from pathlib import Path
from functools import partial

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.agent_cfg  = self.agent._config
        metric_cache = MetricCacheLoader(Path(self.agent_cfg.train_metric_cache_path))

        self.train_metric_cache_paths = metric_cache.metric_cache_paths
        self.test_metric_cache_paths = metric_cache.metric_cache_paths
        agent_traj_samples = self.agent._trajectory_sampling

        self.get_scores = partial(get_scores, agent_traj_samples)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features)
        loss, loss_dict = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, val in loss_dict.items():
            self.log(f"{logging_prefix}/{name}", val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if logging_prefix == 'val':
            #compute PDMS
            score_res = self.compute_scores(targets,prediction['trajectory'])
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
    
    def compute_scores(self, targets:Dict[str,Any], proposals:Tensor, test=True):
        if self.agent.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths
        
        if self.agent_cfg.pdm_score:
            data_points = [
                    {
                        "token": metric_cache_paths[token],
                        "poses": poses,
                        "test": test
                    }
                    for token, poses in zip(targets["token"], proposals.cpu().numpy())
                ]
        else:
            raise ValueError(f"Only PDM scores are supported. PDM score is set to {self.agent_cfg.pdm_score}")

        
        all_res = self.get_scores(data_points)

        return all_res