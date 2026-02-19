from typing import Any, Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.metrics.compute_metrics import get_scores
from pathlib import Path
from functools import partial
import numpy as np

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
        is_training = (logging_prefix == "train")
        self.log(f"{logging_prefix}/loss", loss, on_step=is_training, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, val in loss_dict.items():
            self.log(f"{logging_prefix}/{name}", val, on_step=is_training, on_epoch=True, prog_bar=True, sync_dist=True)
        if logging_prefix == 'val':
            #compute PDMS
            score_res = self.compute_scores(targets,prediction['trajectory'])
            batch_size = len(targets["token"])
            # Compute ADE at specific time intervals (1s, 2s, 3s, 4s)
            ade_metrics = self.agent.compute_ade(
                prediction["trajectory"], 
                targets["trajectory"], 
                time_intervals=[1.0, 2.0, 4.0]
            )
            metrics = score_res | ade_metrics
            for name, val in metrics.items():
                self.log(f"{logging_prefix}/{name}", val, on_step=False, 
                on_epoch=True, sync_dist=True, batch_size=batch_size)
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
        total_steps = self.trainer.estimated_stepping_batches
        print(f"DEBUG: Total Stepping Batches = {total_steps}", flush=True)
        return self.agent.get_optimizers(total_steps)
    
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
        agg_scores = self.aggregate_scores(all_res)

        return agg_scores
    
    def aggregate_scores(self, all_res: list[dict]) -> dict:
        """
        Aggregate PDM scores from a batch for logging.
        
        Args:
            all_res: List of dicts from compute_scores, length = batch_size
        
        Returns:
            Dict with aggregated metrics ready for wandb logging
        """
        # Stack the scores array from each result: (batch_size, num_metrics)
        scores = np.stack([res["scores"] for res in all_res], axis=0)
        
        # scores shape: (batch_size, 1, num_metrics) -> squeeze to (batch_size, num_metrics)
        if scores.ndim == 3:
            scores = scores.squeeze(1)
        
        # Define metric names (must match the order in compute_metrics.py)
        metric_names = [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            # "driving_direction_compliance",  # uncomment if you added this
            "ego_progress",
            "time_to_collision",
            "comfort",
        ]
        
        # Compute mean for each metric
        aggregated = {}
        for i, name in enumerate(metric_names):
            aggregated[f"{name}"] = scores[:, i].mean()
        

        # Extract actual PDM scores from each result
        pdm_scores = np.array([res["pdm_score"] for res in all_res])
        aggregated["pdm_score"] = pdm_scores.mean()

        # Overall metrics
        aggregated["mean_score"] = scores.mean()
        
        # Optional: track failure rates (useful for multiplier metrics)
        aggregated["collision_rate"] = (scores[:, 0] < 1.0).mean()
        aggregated["off_road_rate"] = (scores[:, 1] < 1.0).mean()
        
        return aggregated