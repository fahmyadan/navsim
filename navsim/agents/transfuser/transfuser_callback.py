from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from PIL import ImageColor
import wandb
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.visualization.config import AGENT_CONFIG, MAP_LAYER_CONFIG


class TransfuserCallback(pl.Callback):
    """Visualization Callback for TransFuser during training."""

    def __init__(
        self,
        config: TransfuserConfig,
        num_plots: int = 3,
        num_rows: int = 2,
        num_columns: int = 2,
    ) -> None:
        """
        Initializes the visualization callback.
        :param config: global config dataclass of TransFuser
        :param num_plots: number of images tiles, defaults to 3
        :param num_rows: number of rows in image tile, defaults to 2
        :param num_columns: number of columns in image tile, defaults to 2
        """

        self._config = config

        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns

    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        device = lightning_module.device
        # Collect all plots and log them in a single call to avoid step conflicts
        images_to_log = {}
        
        for idx_plot in range(self._num_plots):
            features, targets = next(iter(trainer.val_dataloaders))
            features, targets = dict_to_device(features, device), dict_to_device(targets, device)
            with torch.no_grad():
                predictions = lightning_module.agent.forward(features)

            features, targets, predictions = (
                dict_to_device(features, "cpu"),
                dict_to_device(targets, "cpu"),
                dict_to_device(predictions, "cpu"),
            )
            grid = self._visualize_model(features, targets, predictions)
            # Convert grid from (C, H, W) to (H, W, C) for wandb.Image
            # vutils.make_grid returns tensor in channels-first format (C, H, W)
            # wandb.Image expects numpy array in channels-last format (H, W, C)
            grid_numpy = grid.numpy().transpose(1, 2, 0)
            images_to_log[f"val_plot_{idx_plot}"] = wandb.Image(grid_numpy)
        
        # Log all images at once using global_step to match wandb's step counter
        # This ensures images are logged at the same step as validation metrics
        # and avoids step conflicts when logging multiple images
        if images_to_log:
            trainer.logger.experiment.log(images_to_log, step=trainer.global_step, commit=True)
            # trainer.logger.experiment.log(f"val_plot_{idx_plot}", grid, global_step=trainer.current_epoch)

    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Inherited, see superclass."""

        # device = lightning_module.device
        # for idx_plot in range(self._num_plots):
        #     features, targets = next(iter(trainer.train_dataloader))
        #     features, targets = dict_to_device(features, device), dict_to_device(targets, device)
        #     with torch.no_grad():
        #         predictions = lightning_module.agent.forward(features)

        #     features, targets, predictions = (
        #         dict_to_device(features, "cpu"),
        #         dict_to_device(targets, "cpu"),
        #         dict_to_device(predictions, "cpu"),
        #     )
        #     grid = self._visualize_model(features, targets, predictions)
        #     trainer.logger.experiment.add_image(f"train_plot_{idx_plot}", grid, global_step=trainer.current_epoch)

    def _visualize_model(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Create tile of input-output visualizations for TransFuser.
        :param features: dictionary of feature names and tensors
        :param targets: dictionary of target names and tensors
        :param predictions: dictionary of target names and predicted tensors
        :return: image tiles as RGB tensors
        """
        camera = features["camera_feature"].permute(0, 2, 3, 1).numpy()
        bev = targets["bev_semantic_map"].numpy()
        # lidar_map = features["lidar_feature"].squeeze(1).numpy()
        agent_labels = targets["agent_labels"].numpy()
        agent_states = targets["agent_states"].numpy()
        trajectory = targets["trajectory"].numpy()

        pred_bev = predictions["bev_semantic_map"].argmax(1).numpy()
        pred_agent_labels = predictions["agent_labels"].sigmoid().numpy()
        pred_agent_states = predictions["agent_states"].numpy()
        pred_trajectory = predictions["trajectory"].numpy()

        plots = []
        for sample_idx in range(self._num_rows * self._num_columns):
            plot = np.zeros((256, 768, 3), dtype=np.uint8)
            plot[:128, :512] = (camera[sample_idx] * 255).astype(np.uint8)[::2, ::2]

            # plot[128:, :256] = semantic_map_to_rgb(bev[sample_idx], self._config)
            # plot[128:, 256:512] = semantic_map_to_rgb(pred_bev[sample_idx], self._config)
            gt_bev_rgb = semantic_map_to_rgb(bev[sample_idx], self._config)
            pred_bev_rgb = semantic_map_to_rgb(pred_bev[sample_idx], self._config)

            gt_valid_mask = agent_labels[sample_idx] > 0.5
            # agent_states_ = agent_states[sample_idx][agent_labels[sample_idx]]
            agent_states_ = agent_states[sample_idx][gt_valid_mask]
            pred_agent_states_ = pred_agent_states[sample_idx][pred_agent_labels[sample_idx] > 0.5]
            # plot[:, 512:] = lidar_map_to_rgb(
            #     lidar_map[sample_idx],
            #     agent_states_,
            #     pred_agent_states_,
            #     trajectory[sample_idx],
            #     pred_trajectory[sample_idx],
            #     self._config,
            # )
            gt_bev_rgb = trajectory_on_bev_to_rgb(
                gt_bev_rgb,
                trajectory[sample_idx],
                # pred_trajectory[sample_idx],
                np.array([]),
                agent_states_,
                np.array([]),
                # pred_agent_states_,
                self._config,
            )
            pred_bev_rgb = trajectory_on_bev_to_rgb(
                pred_bev_rgb,
                trajectory[sample_idx],
                pred_trajectory[sample_idx],
                agent_states_,
                pred_agent_states_,
                self._config,
            )

            plot[128:, :256] = gt_bev_rgb
            plot[128:, 256:512] = pred_bev_rgb

            # Add text labels to distinguish GT and predicted
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background for text
            
            # Add "Camera" label
            text = "Camera"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(plot, (5, 5), (text_width + 10, text_height + baseline + 10), bg_color, -1)
            cv2.putText(plot, text, (10, text_height + 10), font, font_scale, color, thickness)
            
            # Add "GT BEV" label
            text = "GT BEV"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(plot, (5, 133), (text_width + 10, 133 + text_height + baseline + 10), bg_color, -1)
            cv2.putText(plot, text, (10, 133 + text_height + 10), font, font_scale, color, thickness)
            
            # Add "Pred BEV" label
            text = "Pred BEV"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(plot, (261, 133), (261 + text_width + 10, 133 + text_height + baseline + 10), bg_color, -1)
            cv2.putText(plot, text, (266, 133 + text_height + 10), font, font_scale, color, thickness)

            plots.append(torch.tensor(plot).permute(2, 0, 1))

        return vutils.make_grid(plots, normalize=False, nrow=self._num_rows)


def dict_to_device(dict: Dict[str, torch.Tensor], device: Union[torch.device, str]) -> Dict[str, torch.Tensor]:
    """
    Helper function to move tensors from dictionary to device.
    :param dict: dictionary of names and tensors
    :param device: torch device to move tensors to
    :return: dictionary with tensors on specified device
    """
    for key in dict.keys():
        dict[key] = dict[key].to(device) if isinstance(dict[key], torch.Tensor) else dict[key]
    return dict


def semantic_map_to_rgb(semantic_map: npt.NDArray[np.int64], config: TransfuserConfig) -> npt.NDArray[np.uint8]:
    """
    Convert semantic map to RGB image.
    :param semantic_map: numpy array of segmentation map (multi-channel)
    :param config: global config dataclass of TransFuser
    :return: RGB image as numpy array
    """

    height, width = semantic_map.shape[:2]
    rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    for label in range(1, config.num_bev_classes):

        if config.bev_semantic_classes[label][0] == "linestring":
            hex_color = MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]["line_color"]
        else:
            layer = config.bev_semantic_classes[label][-1][0]  # take color of first element
            hex_color = (
                AGENT_CONFIG[layer]["fill_color"]
                if layer in AGENT_CONFIG.keys()
                else MAP_LAYER_CONFIG[layer]["fill_color"]
            )

        rgb_map[semantic_map == label] = ImageColor.getcolor(hex_color, "RGB")
    return rgb_map[::-1, ::-1]


def lidar_map_to_rgb(
    lidar_map: npt.NDArray[np.int64],
    agent_states: npt.NDArray[np.float32],
    pred_agent_states: npt.NDArray[np.float32],
    trajectory: npt.NDArray[np.float32],
    pred_trajectory: npt.NDArray[np.float32],
    config: TransfuserConfig,
) -> npt.NDArray[np.uint8]:
    """
    Converts lidar histogram map with predictions and targets to RGB.
    :param lidar_map: lidar histogram raster
    :param agent_states: target agent bounding box states
    :param pred_agent_states: predicted agent bounding box states
    :param trajectory: target trajectory of human operator (ego GT trajectory)
    :param pred_trajectory: predicted trajectory of agent (ego predicted trajectory)
    :param config: global config dataclass of TransFuser
    :return: RGB image for training visualization
    """
    # Ego trajectory colors: GT is RED, predicted is BLUE
    gt_color, pred_color = (255, 0, 0), (0, 0, 255)  # RED for GT ego, BLUE for predicted ego
    point_size = 4

    height, width = lidar_map.shape[:2]

    def coords_to_pixel(coords):
        """Convert local coordinates to pixel indices."""
        pixel_center = np.array([[height / 2.0, width / 2.0]])
        coords_idcs = (coords / config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)

    rgb_map = (lidar_map * 255).astype(np.uint8)
    rgb_map = 255 - rgb_map[..., None].repeat(3, axis=-1)

    for color, agent_state_array in zip([gt_color, pred_color], [agent_states, pred_agent_states]):
        for agent_state in agent_state_array:
            agent_box = OrientedBox(
                StateSE2(*agent_state[BoundingBox2DIndex.STATE_SE2]),
                agent_state[BoundingBox2DIndex.LENGTH],
                agent_state[BoundingBox2DIndex.WIDTH],
                1.0,
            )
            exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
            exterior = coords_to_pixel(exterior)
            exterior = np.flip(exterior, axis=-1)
            cv2.polylines(rgb_map, [exterior], isClosed=True, color=color, thickness=2)

    for color, traj in zip([gt_color, pred_color], [trajectory, pred_trajectory]):
        trajectory_indices = coords_to_pixel(traj[:, :2])
        for x, y in trajectory_indices:
            cv2.circle(rgb_map, (y, x), point_size, color, -1)  # -1 fills the circle

    return rgb_map[::-1, ::-1]

# def trajectory_on_bev_to_rgb(
#     bev_rgb_map: npt.NDArray[np.uint8],
#     gt_trajectory: npt.NDArray[np.float32],
#     pred_trajectory: npt.NDArray[np.float32],
#     gt_agent_states: npt.NDArray[np.float32],
#     pred_agent_states: npt.NDArray[np.float32],
#     config: TransfuserConfig,
# ) -> npt.NDArray[np.uint8]:
#     """
#     Overlays GT and predicted trajectories and agent bounding boxes on BEV semantic map.
#     :param bev_rgb_map: RGB image of BEV semantic map (from semantic_map_to_rgb)
#     :param gt_trajectory: Ground truth ego trajectory in local coordinates (N, 3) [x, y, heading]
#     :param pred_trajectory: Predicted ego trajectory in local coordinates (N, 3) [x, y, heading]
#     :param gt_agent_states: Ground truth agent bounding box states in local coordinates (M, 5) [x, y, heading, length, width]
#     :param pred_agent_states: Predicted agent bounding box states in local coordinates (K, 5) [x, y, heading, length, width]
#     :param config: TransfuserConfig with BEV parameters
#     :return: RGB image with trajectories and bounding boxes overlaid
#     """
#     # Make a copy to avoid modifying the original
#     rgb_map = bev_rgb_map.copy()
    
#     # Trajectory colors: GT is RED, predicted is BLUE
#     gt_traj_color, pred_traj_color = (255, 0, 0), (0, 0, 255)  # RED for GT, BLUE for predicted
#     # Agent bounding box colors: GT is GREEN, predicted is YELLOW
#     gt_agent_color, pred_agent_color = (0, 255, 0), (0, 255, 255)  # GREEN for GT, YELLOW for predicted
#     traj_point_size = 3
#     agent_box_thickness = 2
    
#     height, width = rgb_map.shape[:2]
    
#     def coords_to_pixel_bev(coords):
#         """
#         Convert local coordinates to pixel indices for BEV semantic map.
        
#         The BEV semantic map is created with ego at [row=0, col=width/2] in the original coordinate system.
#         semantic_map_to_rgb then flips it: rgb_map[::-1, ::-1]
        
#         After the flip:
#         - Row flip: row i -> row (height-1-i)
#         - Col flip: col j -> col (width-1-j)
        
#         So if ego was at [0, width/2] originally, after flip it's at [height-1, width-1-width/2]
#         But we want ego to appear at top center, so we need to transform coordinates accordingly.
#         """
#         # Compute coordinates in original (unflipped) system
#         # Ego is at [0, width/2] in original system
#         pixel_center_original = np.array([[0, width / 2.0]])
#         coords_original = (coords / config.bev_pixel_size) + pixel_center_original
        
#         # Transform to flipped coordinate system
#         # rgb_map[::-1, ::-1] means: new_row = height-1-old_row, new_col = width-1-old_col
#         coords_flipped = np.zeros_like(coords_original)
#         coords_flipped[:, 0] = height - 1 - coords_original[:, 0]  # Row flip
#         coords_flipped[:, 1] = width - 1 - coords_original[:, 1]   # Col flip
        
#         return coords_flipped.astype(np.int32)
    
#     # Draw agent bounding boxes
#     for color, agent_state_array in zip([gt_agent_color, pred_agent_color], [gt_agent_states, pred_agent_states]):
#         if len(agent_state_array) == 0:
#             continue
#         for agent_state in agent_state_array:
#             agent_box = OrientedBox(
#                 StateSE2(*agent_state[BoundingBox2DIndex.STATE_SE2]),
#                 agent_state[BoundingBox2DIndex.LENGTH],
#                 agent_state[BoundingBox2DIndex.WIDTH],
#                 1.0,
#             )
#             exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
#             exterior = coords_to_pixel_bev(exterior)
#             # Convert [row, col] to [col, row] for cv2 (cv2 uses x=col, y=row)
#             exterior_cv2 = np.flip(exterior, axis=-1)
#             cv2.polylines(rgb_map, [exterior_cv2], isClosed=True, color=color, thickness=agent_box_thickness)
    
#     # Draw trajectories
#     for color, traj in zip([gt_traj_color, pred_traj_color], [gt_trajectory, pred_trajectory]):
#         if len(traj) == 0:
#             continue
#         # Extract x, y coordinates (first two columns)
#         trajectory_coords = traj[:, :2]
#         trajectory_indices = coords_to_pixel_bev(trajectory_coords)
#         # Convert [row, col] to [col, row] for cv2
#         trajectory_indices_cv2 = np.flip(trajectory_indices, axis=-1)
#         for point in trajectory_indices_cv2:
#             x, y = point[0], point[1]
#             # Check bounds before drawing
#             if 0 <= x < width and 0 <= y < height:
#                 cv2.circle(rgb_map, (x, y), traj_point_size, color, -1)  # -1 fills the circle
    
#     return rgb_map

def trajectory_on_bev_to_rgb(
    bev_rgb_map: npt.NDArray[np.uint8],
    gt_trajectory: npt.NDArray[np.float32],
    pred_trajectory: npt.NDArray[np.float32],
    gt_agent_states: npt.NDArray[np.float32],
    pred_agent_states: npt.NDArray[np.float32],
    config: TransfuserConfig,
) -> npt.NDArray[np.uint8]:
    """
    Overlays GT and predicted trajectories and agent bounding boxes on BEV semantic map.
    """
    # 1. UN-FLIP the incoming map so we are drawing in the true coordinate space
    rgb_map = bev_rgb_map[::-1, ::-1].copy()
    
    gt_traj_color, pred_traj_color = (255, 0, 0), (0, 0, 255)  # RED GT, BLUE Pred
    gt_agent_color, pred_agent_color = (0, 255, 0), (0, 255, 255)  # GREEN GT, YELLOW Pred
    traj_point_size = 3
    agent_box_thickness = 2
    
    height, width = rgb_map.shape[:2]
    
    def coords_to_pixel_bev(coords):
        # 2. FIX ORIGIN: Ego is in the center of the spatial grid
        pixel_center = np.array([[height / 2.0, width / 2.0]])
        coords_idcs = (coords / config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)
    
    ego_color = (255, 255, 255)  # WHITE for Ego vehicle
    ego_length = 4.08  # Standard nuPlan/Navsim ego vehicle length in meters
    ego_width = 1.73   # Standard nuPlan/Navsim ego vehicle width in meters
    
    ego_box = OrientedBox(
        StateSE2(0.0, 0.0, 0.0),
        ego_length,
        ego_width,
        1.0,
    )
    ego_exterior = np.array(ego_box.geometry.exterior.coords).reshape((-1, 1, 2))
    ego_exterior_pixels = coords_to_pixel_bev(ego_exterior)
    ego_exterior_cv2 = np.flip(ego_exterior_pixels, axis=-1)  # [row, col] -> [x, y]
    
    # Fill the ego box so it stands out strongly against the map
    cv2.fillPoly(rgb_map, [ego_exterior_cv2], color=ego_color)
    cv2.polylines(rgb_map, [ego_exterior_cv2], isClosed=True, color=(0, 0, 0), thickness=1)
    # Draw agent bounding boxes
    for color, agent_state_array in zip([gt_agent_color, pred_agent_color], [gt_agent_states, pred_agent_states]):
        if len(agent_state_array) == 0:
            continue
        for agent_state in agent_state_array:
            agent_box = OrientedBox(
                StateSE2(*agent_state[BoundingBox2DIndex.STATE_SE2]),
                agent_state[BoundingBox2DIndex.LENGTH],
                agent_state[BoundingBox2DIndex.WIDTH],
                1.0,
            )
            exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
            exterior = coords_to_pixel_bev(exterior)
            exterior_cv2 = np.flip(exterior, axis=-1)  # Convert [row, col] to [col, row] for cv2
            cv2.polylines(rgb_map, [exterior_cv2], isClosed=True, color=color, thickness=agent_box_thickness)
    
    # Draw trajectories
    for color, traj in zip([gt_traj_color, pred_traj_color], [gt_trajectory, pred_trajectory]):
        if len(traj) == 0:
            continue
        trajectory_coords = traj[:, :2]
        trajectory_indices = coords_to_pixel_bev(trajectory_coords)
        trajectory_indices_cv2 = np.flip(trajectory_indices, axis=-1)
        
        for point in trajectory_indices_cv2:
            x, y = point[0], point[1]
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(rgb_map, (x, y), traj_point_size, color, -1)
    
    # 3. RE-FLIP the final image so the ego vehicle faces "up" for human viewing
    return rgb_map[::-1, ::-1]