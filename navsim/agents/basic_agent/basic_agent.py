import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np 
import cv2
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.ego_status_mlp_agent import TrajectoryTargetBuilder
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig
from dataclasses import dataclass

@dataclass
class BasicAgentConfig:

    img_encoder: str = "resnet34"
    num_history_steps: int = 2
    lr :float = 1e-4
    train_metric_cache_path: str = None 
    pdm_score: bool = True

class BasicModel(nn.Module):

    def __init__(self,config) -> None:
        super().__init__()
        
        self.config = config

        if self.config.img_encoder == "resnet34":
            self.img_encoder = torchvision.models.resnet34() 
        else:
            raise ValueError(f"Only support resnet34 encoder. Got {self.config.img_encoder}")
        
      




class BasicAgent(AbstractAgent):

    def __init__(self, trajectory_sampling, config):
        super().__init__(trajectory_sampling)

        self.model = BasicModel(config)
        self._lr = config.lr
        
    
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
        
        return [TrajectoryTargetBuilder(self._trajectory_sampling)]
    
    def get_feature_builders(self):

        return [BasicAgentFeatureBuilder()]

    def compute_loss(self, features, targets, predictions) -> torch.Tensor:
        """Implement l2 loss for pred traj and expert
        """
        return 0
    
    def get_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        return optimizer

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
        f0 = cameras.cam_f0.image[28:-28]
        tensor_image = transforms.ToTensor()(f0)

        return tensor_image