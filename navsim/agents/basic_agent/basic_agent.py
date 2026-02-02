import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np 
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.ego_status_mlp_agent import TrajectoryTargetBuilder
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.common.enums import StateSE2Index
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, List, Optional

@dataclass
class BasicAgentConfig:

    img_encoder: str = "resnet34"
    num_history_steps: int = 2
    lr: float = 1e-4
    hidden_dim: int = 512  # Hidden dimension for trajectory MLP
    train_metric_cache_path: str = None 
    pdm_score: bool = True
    stage: str = "train"

class BasicModel(nn.Module):

    def __init__(self, config: BasicAgentConfig, trajectory_sampling: TrajectorySampling) -> None:
        super().__init__()
        
        self.config = config
        self._num_poses = trajectory_sampling.num_poses  # e.g., 8 for 4s horizon at 0.5s intervals

        # Build image encoder (ResNet34 backbone)
        if self.config.img_encoder == "resnet34":
            resnet = torchvision.models.resnet34(weights=None) 
            # Remove last fc layer, keep avgpool to get (batch, 512) features
            exclude = ['fc']
            modules = [(name, m) for name, m in resnet.named_children() if name not in exclude]
            self.feat_extractor = nn.Sequential(OrderedDict(modules))
            img_feat_dim = 512  # ResNet34 outputs 512 channels after avgpool
        else:
            raise ValueError(f"Only support resnet34 encoder. Got {self.config.img_encoder}")
        
        # Ego status feature dimension: driving_command(4) + velocity(2) + acceleration(2) = 8
        status_feat_dim = 8
        
        # Total feature dimension after concatenation
        total_feat_dim = img_feat_dim + status_feat_dim
        
        # Trajectory prediction MLP head
        # Output: num_poses * 3 (x, y, heading for each pose)
        self.trajectory_head = nn.Sequential(
            nn.Linear(total_feat_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self._num_poses * StateSE2Index.size()),  # num_poses * 3
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        img = features['camera_feature']  # (B, 3, H, W)
        ego_status = features['status_feature']  # (B, 8)
        
        B = img.shape[0]
        
        # Encode image features through ResNet -> (B, 512, H', W') -> avgpool -> (B, 512)
        img_feats = self.feat_extractor(img)
        img_feats = img_feats.reshape(B, -1)  # Flatten to (B, 512)

        # Concatenate image features with ego status -> (B, 512 + 8)
        combined_feats = torch.cat([img_feats, ego_status], dim=-1)

        # Predict trajectory through MLP head -> (B, num_poses * 3)
        trajectory_flat = self.trajectory_head(combined_feats)
        
        # Reshape to (B, num_poses, 3) where 3 = (x, y, heading)
        trajectory = trajectory_flat.reshape(B, self._num_poses, StateSE2Index.size())
        
        # Constrain heading to [-pi, pi] using tanh -> tanh(x) [-1,1]
        trajectory[..., StateSE2Index.HEADING] = trajectory[..., StateSE2Index.HEADING].tanh() * np.pi

        return {"trajectory": trajectory}

        
      




class BasicAgent(AbstractAgent):

    def __init__(self, trajectory_sampling: TrajectorySampling, config: BasicAgentConfig):
        super().__init__(trajectory_sampling)

        # Pass trajectory_sampling to model so it knows num_poses
        self.model = BasicModel(config, trajectory_sampling)
        self._lr = config.lr
        self._config = config
        
    
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__
    
    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc= False,
        )
    
    def get_target_builders(self):
        
        return [BasicAgentTargetBuilder(self._trajectory_sampling)]
    
    def get_feature_builders(self):

        return [BasicAgentFeatureBuilder()]

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute L1 loss between predicted and ground-truth trajectories.
        
        :param features: Input features (not used for loss computation)
        :param targets: Ground-truth targets containing 'trajectory' key
        :param predictions: Model predictions containing 'trajectory' key
        :return: Tuple of (total_loss, loss_dict for logging)
        """
        pred_trajectory = predictions["trajectory"]  # (B, num_poses, 3)
        gt_trajectory = targets["trajectory"]  # (B, num_poses, 3)
        
        # L1 loss for trajectory prediction
        trajectory_loss = torch.nn.functional.l1_loss(pred_trajectory, gt_trajectory)
        
        loss_dict = {
            "trajectory_loss": trajectory_loss,
        }
        
        # Compute ADE at specific time intervals (1s, 2s, 3s, 4s)
        ade_metrics = self.compute_ade(
            pred_trajectory, 
            gt_trajectory, 
            time_intervals=[1.0, 2.0, 4.0]
        )
        loss_dict.update(ade_metrics)
        
        return trajectory_loss , loss_dict
    
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
    
    def get_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        return optimizer

    def forward(self,features):


        return self.model(features)


class BasicAgentFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput):
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)

        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )
    
        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        f0 = cameras.cam_f0.image[28:-28] # (3,1024,1920)
        tensor_image = transforms.ToTensor()(f0)

        return tensor_image
    

class BasicAgentTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        
        return {"trajectory": torch.tensor(future_trajectory.poses),
                "token": scene.scene_metadata.initial_token}