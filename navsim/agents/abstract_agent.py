from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene
        self._trajectory_sampling = trajectory_sampling

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses, self._trajectory_sampling)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self, estimate_stepping_batches: int, 
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []

    def compute_ade(
        self,
        pred_trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        time_intervals: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Average Displacement Error (ADE) at specified time intervals.
        
        ADE measures the average L2 distance between predicted and ground-truth 
        (x, y) positions at the specified time points.
        
        :param pred_trajectory: Predicted trajectory (B, num_poses, 3) where 3 = (x, y, heading)
        :param gt_trajectory: Ground-truth trajectory (B, num_poses, 3)
        :param time_intervals: List of time points in seconds to evaluate ADE.
                              If None, uses all poses. 
                              Example: [1.0, 2.0, 3.0, 4.0] for 1s, 2s, 3s, 4s
        :return: Dict with 'ade' (overall) and 'ade_at_Xs' for each time interval
        """
        interval_length = self._trajectory_sampling.interval_length  # e.g., 0.5s
        num_poses = self._trajectory_sampling.num_poses
        
        # Extract (x, y) positions only (ignore heading)
        pred_xy = pred_trajectory[..., :2]  # (B, num_poses, 2)
        gt_xy = gt_trajectory[..., :2]  # (B, num_poses, 2)
        
        # Compute L2 displacement at each pose
        displacement = torch.norm(pred_xy - gt_xy, dim=-1)  # (B, num_poses)
        
        ade_dict = {}
        
        if time_intervals is None:
            # ADE over all poses
            ade_dict["ade"] = displacement.mean()
        else:
            # Convert time intervals to pose indices
            # Pose index = time / interval_length - 1 (since pose 0 is at interval_length seconds)
            # e.g., with interval_length=0.5: time=1.0s -> index 1, time=2.0s -> index 3
            pose_indices = []
            for t in time_intervals:
                # Index = (t / interval_length) - 1, but we need to handle the offset
                # Pose 0 corresponds to time = interval_length
                # Pose i corresponds to time = (i + 1) * interval_length
                idx = int(t / interval_length) - 1
                if 0 <= idx < num_poses:
                    pose_indices.append(idx)
                    # ADE at this specific time
                    ade_dict[f"ade_at_{t:.1f}s"] = displacement[:, idx].mean()
            
            if pose_indices:
                # Overall ADE at specified intervals
                selected_displacement = displacement[:, pose_indices]  # (B, len(pose_indices))
                ade_dict["ade"] = selected_displacement.mean()
            else:
                # Fallback to all poses if no valid intervals
                ade_dict["ade"] = displacement.mean()
        
        return ade_dict