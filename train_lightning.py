# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import hydra

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from scene import get_data_loader, initialize_scene_info


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.scene.model_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.scene.model_path, "wandb"), exist_ok=True)

    scene_info = initialize_scene_info(cfg.scene)

    if "3dgs" in cfg.train_model:
        from model.vanilla_gsplat import VanillaGSplat

        print("Initialize training for 3DGS")

        module = VanillaGSplat(cfg=cfg, scene_info=scene_info)
    elif "2dgs" in cfg.train_model:
        from model.GS2D_gsplat import Gaussians2D

        print("Initialize training for 2DGS")

        module = Gaussians2D(cfg=cfg, scene_info=scene_info)
    else:
        raise RuntimeError(f"cannot recognize the train model {cfg.train_model}")

    if cfg.wandb.use_wandb:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=str(cfg.exp_name),
            save_dir=cfg.scene.model_path,
        )
        logger.log_hyperparams(cfg)
    else:
        logger = None

    if cfg.opt.depth_loss:
        assert (
            cfg.opt.batch_size == 1
        ), "depth loss does not support batch size > 1 yet!"

    train_loader = get_data_loader(
        scene_info.train_cameras,
        batch_size=cfg.opt.batch_size,
        subset="train",
        shuffle=cfg.train.shuffle,
    )
    valid_loader = get_data_loader(
        scene_info.valid_cameras, subset="valid", shuffle=False
    )
    test_loader = get_data_loader(scene_info.test_cameras, subset="test", shuffle=False)

    factor = cfg.opt.steps_scaler
    max_steps = int(cfg.opt.iterations * factor)
    print(f"max iterations: {max_steps}")

    trainer = L.Trainer(
        max_steps=max_steps,
        logger=logger,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        devices=1,  # currently not supporting multi-gpu training yet
    )

    trainer.fit(
        model=module,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # Evaluation
    trainer.test(
        model=module,
        dataloaders=test_loader,
    )


if __name__ == "__main__":
    main()
