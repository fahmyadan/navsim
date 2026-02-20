import logging
from pathlib import Path
from typing import Tuple
import os
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tensorflow.python.eager.context import check_alive
from torch.utils.data import DataLoader
from datetime import datetime
import glob
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.callbacks.wandb_callback import LogConfigCallback
import re
import json
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


# class SaveMetadataCallback(pl.Callback):
#     """Callback to save job metadata after WandB is initialized."""
#     def __init__(self, metadata_dir, checkpoint_dir, experiment_name, agent_name, 
#                  wandb_project, wandb_entity, timestamp, slurm_job_id):
#         self.metadata_dir = metadata_dir
#         self.checkpoint_dir = checkpoint_dir
#         self.experiment_name = experiment_name
#         self.agent_name = agent_name
#         self.wandb_project = wandb_project
#         self.wandb_entity = wandb_entity
#         self.timestamp = timestamp
#         self.slurm_job_id = slurm_job_id
#         self.saved = False
    
#     def on_train_start(self, trainer, pl_module):
#         """Save metadata once WandB run is confirmed initialized."""
#         if not self.saved and trainer.logger and hasattr(trainer.logger, 'experiment'):
#             wandb_run_id = trainer.logger.experiment.id if trainer.logger.experiment else None
#             if wandb_run_id:
#                 save_job_metadata(
#                     metadata_dir=self.metadata_dir,
#                     checkpoint_dir=self.checkpoint_dir,
#                     experiment_name=self.experiment_name,
#                     agent_name=self.agent_name,
#                     wandb_run_id=wandb_run_id,
#                     wandb_project=self.wandb_project,
#                     wandb_entity=self.wandb_entity,
#                     timestamp=self.timestamp,
#                     slurm_job_id=self.slurm_job_id,
#                 )
#                 logger.info(f"Saved WandB run ID {wandb_run_id} to metadata")
#                 self.saved = True
# --- UPDATED CALLBACK ---
# --- HELPER: GET RANK ---
def get_rank():
    # Works for Slurm (srun) and TorchElastic
    return int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))

class UpdateMetadataCallback(pl.Callback):
    """Callback to UPDATE existing metadata with WandB ID on train start."""
    def __init__(self, metadata_dir, slurm_job_id):
        self.metadata_dir = metadata_dir
        self.slurm_job_id = slurm_job_id
        self.saved = False
    
    def on_train_start(self, trainer, pl_module):
        # [FIX ISSUE 1] Only Rank 0 updates metadata
        if trainer.global_rank == 0 and not self.saved and trainer.logger:
            experiment = getattr(trainer.logger, 'experiment', None)
            wandb_run_id = getattr(experiment, 'id', None)
            
            if wandb_run_id:
                try:
                    save_job_metadata(
                        metadata_dir=self.metadata_dir,
                        slurm_job_id=self.slurm_job_id,
                        wandb_run_id=wandb_run_id
                    )
                    self.saved = True
                except Exception as e:
                    logger.error(f"Failed to update metadata with WandB ID: {e}")


CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    # now = datetime.now().strftime("%H:%M-%d-%m-%Y")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )
    rank = get_rank()

    # --- SLURM SETUP ---
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    is_requeued = slurm_restart_count > 0
    metadata_dir = os.path.join(cfg.output_dir, ".job_metadata")

    # --- SYNCHRONIZED INITIALIZATION ---
    metadata = None

    if rank == 0 and not is_requeued:
        # --- RANK 0: LEADER ---
        # 1. Define the 'Truth' (timestamp, paths)
        now = datetime.now().strftime("%H:%M-%d-%m-%Y")
        
        checkpoint_callback = instantiate(cfg.checkpoint)
        checkpoint_callback.dirpath = os.path.join(checkpoint_callback.dirpath, agent.name(), 
                                                cfg.experiment_name, now, "checkpoints")
        checkpoint_callback.filename = "model_" + checkpoint_callback.filename

        # 2. Write Metadata immediately
        if slurm_job_id:
            save_job_metadata(
                metadata_dir=metadata_dir,
                slurm_job_id=slurm_job_id,
                checkpoint_dir=checkpoint_callback.dirpath,
                experiment_name=cfg.experiment_name,
                timestamp=now,
                wandb_run_id=None
            )
            # Load it back to ensure consistency variable
            # After saving, load it back to get complete metadata
            metadata = load_job_metadata_by_id(metadata_dir, slurm_job_id)
            if metadata is None:
                raise ValueError(f"Failed to load metadata after saving for {slurm_job_id}")

    # --- ALL RANKS: FOLLOWER LOGIC ---
    # Ranks > 0 (or Rank 0 if requeued) must wait for the file to exist
    if metadata is None: # This is True for Rank > 0 OR Rank 0 (requeued)
        if slurm_job_id is None:
             raise ValueError("SLURM_JOB_ID missing.")
             
        metadata_path = os.path.join(metadata_dir, f"{slurm_job_id}.json")
        
        # [FIX ISSUE 1] Spin-wait until Rank 0 writes the file
        logger.info(f"Rank {rank}: Waiting for metadata at {metadata_path}...")
        max_wait = 300  # 5 minutes max
        waited = 0
        while not os.path.exists(metadata_path) and waited < max_wait:
            time.sleep(1)
            waited += 1
            
        if not os.path.exists(metadata_path):
            raise TimeoutError(f"Rank {rank}: Metadata file not created by rank 0 after {max_wait}s")
                    
        # Optional: extra sleep to ensure write flush
        time.sleep(0.5) 
        
        # Load the 'Truth' generated by Rank 0
        metadata = load_job_metadata_by_id(metadata_dir, slurm_job_id)
        if metadata is None:
            raise ValueError(f"Metadata corrupted or missing for {slurm_job_id}")

    # --- UNIFIED CONFIGURATION ---
    # Now ALL ranks have the EXACT same path and timestamp
    checkpoint_dir = metadata["checkpoint_dir"]
    checkpoint_callback = instantiate(cfg.checkpoint)
    checkpoint_callback.dirpath = checkpoint_dir
    checkpoint_callback.filename = "model_" + checkpoint_callback.filename



    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    # --- TRAINER & WANDB SETUP ---
    
    # 1. Determine Resume Checkpoint
    # resume_ckpt = find_resume_checkpoint(checkpoint_callback.dirpath, is_requeued)
    resume_ckpt = None
    if is_requeued:
        if not os.path.exists(checkpoint_callback.dirpath):
            raise RuntimeError(
                f"Job requeued but checkpoint directory does not exist: {checkpoint_callback.dirpath}. "
                f"This suggests the job was preempted before any checkpoints were saved."
            )
        resume_ckpt = find_resume_checkpoint(checkpoint_callback.dirpath, is_requeued)
        if resume_ckpt:
            logger.info(f"Job requeued: Auto-detected checkpoint for resume: {resume_ckpt}")
        else:
            logger.warning(f"Job requeued but no checkpoint found in {checkpoint_callback.dirpath}")
    else:
        if hasattr(cfg, 'resume_from_checkpoint') and cfg.resume_from_checkpoint is not None:
            resume_ckpt = cfg.resume_from_checkpoint
            if not os.path.exists(resume_ckpt):
                raise FileNotFoundError(f"Checkpoint not found: {resume_ckpt}")
            logger.info(f"First run: Resuming from explicit checkpoint: {resume_ckpt}")
        else:
            # No explicit checkpoint: try auto-detection (useful for continuing existing experiments)
            resume_ckpt = find_resume_checkpoint(checkpoint_callback.dirpath, is_requeued)
            if resume_ckpt:
                logger.info(f"First run: Auto-detected checkpoint for resume: {resume_ckpt}")
            else:
                logger.info("First run: Starting training from scratch")
    # 2. Configure Logger
    wandb_logger = None
    callbacks = [*agent.get_training_callbacks(), checkpoint_callback]
    
    # WandB Logic (Only Rank 0 usually logs, but PL handles this)
    if "debug" not in cfg.experiment_name:
        wandb_resume_id = metadata.get("wandb_run_id")
        
        # If we have an ID, we force resume. 
        # Note: If wandb_resume_id is None (First Run), Rank > 0 will see None.
        # PL handles DDP logging (only Rank 0 logs to WandB backend), so this is safe.
        
        wandb_conf = OmegaConf.to_container(cfg.trainer.logger, resolve=True)
        if wandb_resume_id:
             wandb_conf["id"] = wandb_resume_id
             wandb_conf["resume"] = "must"
             
        wandb_logger = instantiate(DictConfig(wandb_conf))
        
        # [FIX ISSUE 1] Only add callback if we don't have an ID yet
        if not wandb_resume_id and rank == 0:
             callbacks.append(UpdateMetadataCallback(metadata_dir, slurm_job_id))
             callbacks.append(LogConfigCallback(cfg))
    else: 
        #no logging 
        wandb_callback = None
        wandb_logger = None 
    
    trainer = pl.Trainer(**cfg.trainer.params, 
                        callbacks=callbacks, 
                        logger=wandb_logger)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path= resume_ckpt
    )

def find_resume_checkpoint(checkpoint_dir, slurm_requeue, prefer_last=True):
    """
    Find checkpoint to resume from.
    Priority: last.ckpt > latest epoch checkpoint > None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Option 1: Use last.ckpt if it exists and prefer_last=True
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if prefer_last and os.path.exists(last_ckpt):
        return last_ckpt
    
    # Option 2: Find latest epoch checkpoint
    pattern = os.path.join(checkpoint_dir, "*_epoch=*.ckpt")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        def get_epoch(path):
            match = re.search(r'epoch=(\d+)', os.path.basename(path))
            return int(match.group(1)) if match else -1
        return max(checkpoints, key=get_epoch)
    
    # Option 3: Fallback to last.ckpt even if prefer_last=False
    if os.path.exists(last_ckpt):
        return last_ckpt
    
    return None
    
# --- HELPER: METADATA I/O ---
def save_job_metadata(metadata_dir, slurm_job_id, **kwargs):
    # [FIX ISSUE 1 & 4] Only Rank 0 writes to disk
    if get_rank() != 0:
        return None
        
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_file = os.path.join(metadata_dir, f"{slurm_job_id}.json")
    
    # [FIX ISSUE 1] Atomic-ish Read-Modify-Write
    data = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass 

    data.update(kwargs)
    data["slurm_job_id"] = slurm_job_id
    
    # Write to a temp file then rename (Atomic write)
    temp_file = metadata_file + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=2)
    os.rename(temp_file, metadata_file)
    
    logger.info(f"Updated metadata in {metadata_file}")
    return metadata_file

def load_job_metadata_by_id(
    metadata_dir: str,
    slurm_job_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Load metadata file for a specific SLURM job ID.
    This prevents race conditions when multiple jobs with the same experiment name run concurrently.
    
    Returns:
        Metadata dictionary or None if not found
    """
    if not os.path.exists(metadata_dir) or slurm_job_id is None:
        return None
    
    # Load metadata file directly by job ID
    metadata_file = os.path.join(metadata_dir, f"{slurm_job_id}.json")
    
    if not os.path.exists(metadata_file):
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded job metadata from {metadata_file}")
    return metadata

if __name__ == "__main__":
    main()
