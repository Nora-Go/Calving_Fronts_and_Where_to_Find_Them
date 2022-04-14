from argparse import ArgumentParser
from models.zones_segmentation_model import ZonesUNet
from models.front_segmentation_model import FrontUNet
from data_processing.glacier_zones_data import GlacierZonesDataModule
from data_processing.glacier_front_data import GlacierFrontDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from torchsummary import summary
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import torch
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(hparams, run_number, running_mode):
    if hparams.target_masks == "zones":
        checkpoint_dir = os.path.join('checkpoints', 'zones_segmentation', 'run_' + str(run_number))
        tb_logs_dir = os.path.join('tb_logs', 'zones_segmentation', 'run_' + str(run_number))
        checkpoint_callback = ModelCheckpoint(monitor='avg_metric_validation',
                                              dirpath=checkpoint_dir,
                                              filename='-{epoch:02d}-{avg_metric_validation:.2f}',
                                              mode='max',  # Here: the higher the IoU the better
                                              save_top_k=1)
        early_stop_callback = EarlyStopping(monitor="avg_metric_validation", patience=30,
                                            verbose=False, mode="max", check_finite=True)
        clip_norm = 1.0
    else:
        checkpoint_dir = os.path.join('checkpoints', 'fronts_segmentation', 'run_' + str(run_number))
        tb_logs_dir = os.path.join('tb_logs', 'fronts_segmentation', 'run_' + str(run_number))
        checkpoint_callback = ModelCheckpoint(monitor='avg_loss_validation',
                                              dirpath=checkpoint_dir,
                                              filename='-{epoch:02d}-{avg_loss_validation:.2f}',
                                              mode='min',  # Here: the lower the loss the better
                                              save_top_k=1)
        early_stop_callback = EarlyStopping(monitor="avg_loss_validation", patience=30,
                                            verbose=False, mode="min", check_finite=True)
        clip_norm = 1.0

    logger = TensorBoardLogger(tb_logs_dir, name="log", default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # if we have already trained this model take up the training at the last checkpoint
    if os.path.isfile(checkpoint_dir + "temporary.ckpt"):
        print("Taking up the training where it was left (temporary checkpoint)")
        trainer = Trainer(resume_from_checkpoint=checkpoint_dir + "temporary.ckpt",
                          callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                          deterministic=True,
                          gpus=1,  # Train on gpu
                          gradient_clip_val=clip_norm,
                          logger=logger,
                          max_epochs=hparams.epochs)
    else:
        if running_mode == "batch_overfit":
            # Try to overfit on some batches
            trainer = Trainer.from_argparse_args(hparams,
                                                 callbacks=[checkpoint_callback, lr_monitor],
                                                 deterministic=True,
                                                 fast_dev_run=False,
                                                 flush_logs_every_n_steps=100,
                                                 gpus=1,
                                                 log_every_n_steps=1,
                                                 logger=logger,
                                                 max_epochs=1000,
                                                 overfit_batches=1)
        elif running_mode == "debugging":
            # Debugging mode
            trainer = Trainer.from_argparse_args(hparams,
                                                 callbacks=[checkpoint_callback, lr_monitor],
                                                 deterministic=True,
                                                 fast_dev_run=3,
                                                 flush_logs_every_n_steps=100,
                                                 gpus=1,
                                                 log_every_n_steps=1,
                                                 logger=logger)
        elif running_mode == "training":
            # Training mode
            trainer = Trainer.from_argparse_args(hparams,
                                                 callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                                                 deterministic=True,
                                                 gpus=1,  # Train on gpu
                                                 gradient_clip_val=clip_norm,
                                                 logger=logger,
                                                 max_epochs=hparams.epochs)
        else:
            print("Running mode not recognized")
            sys.exit()

    if hparams.target_masks == "zones":
        datamodule = GlacierZonesDataModule(batch_size=hparams.batch_size,
                                            augmentation=running_mode != "batch_overfit",
                                            parent_dir=hparams.parent_dir,
                                            bright=hparams.bright,
                                            wrap=hparams.wrap,
                                            noise=hparams.noise,
                                            rotate=hparams.rotate,
                                            flip=hparams.flip)
        model = ZonesUNet(vars(hparams))

    else:
        datamodule = GlacierFrontDataModule(batch_size=hparams.batch_size,
                                            augmentation=running_mode != "batch_overfit",
                                            parent_dir=hparams.parent_dir,
                                            bright=hparams.bright,
                                            wrap=hparams.wrap,
                                            noise=hparams.noise,
                                            rotate=hparams.rotate,
                                            flip=hparams.flip)
        model = FrontUNet(vars(hparams))

    summary(model.cuda(), (1, 256, 256))
    print(model.eval())

    trainer.fit(model, datamodule=datamodule)

    # create a checkpoint if we are training (and delete the old one if it exists)
    if running_mode == "training":
        if os.path.isfile(checkpoint_dir + "temporary.ckpt"):
            os.remove(checkpoint_dir + "temporary.ckpt")
        trainer.save_checkpoint(filepath=checkpoint_dir + "temporary.ckpt")


if __name__ == '__main__':
    seed_everything(42)
    torch.multiprocessing.set_start_method('spawn')  # needed if ddp mode in Trainer

    parent_parser = ArgumentParser(add_help=False)
    # add PROGRAM level args
    parent_parser.add_argument('--parent_dir', default=".",
                               help="The directory in which the data directory lies. "
                                    "Default is '.' - this is where the data_preprocessing script has produced it.")
    parent_parser.add_argument('--training_mode', default="training",
                               help="Either 'training', 'debugging' or 'batch_overfit'.")
    parent_parser.add_argument('--epochs', type=int, default=150, help="The number of epochs the model shall be trained."
                                                                       "The weights after the last training epoch will "
                                                                       "be stored in temporary.ckpt."
                                                                       "Train.py will resume training from a "
                                                                       "temporary.ckpt if one is available for this "
                                                                       "run.")
    parent_parser.add_argument('--run_number', type=int, default=5,
                               help="The number how often this model was already trained. "
                                    "If you run train.py twice with the same run_number, "
                                    "the second run will pick up training the first model from the temporary.ckpt.")
    parent_parser.add_argument('--target_masks', default="zones", help="Either 'fronts' or 'zones'. "
                                                                        "This decides which model will be trained.")

    tmp = parent_parser.parse_args()
    if tmp.target_masks == "fronts":
        parser = FrontUNet.add_model_specific_args(parent_parser)
    else:
        parser = ZonesUNet.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    print(vars(hparams))

    assert hparams.target_masks == "fronts" or hparams.target_masks == "zones", \
        "Please set --target_masks correctly. Either 'fronts' or 'zones'."
    assert hparams.training_mode == "training" or hparams.training_mode == "debugging" or hparams.training_mode == "batch_overfit", \
        "Please set --training_mode correctly. Either 'training' or 'debugging' or 'batch_overfit'."

    start_save = time.time()
    main(hparams, hparams.run_number, hparams.training_mode)
    end_save = time.time()
    print(f"Total time for training: {end_save - start_save}")

    # run the following if you want to open tensorboard
    # tensorboard --logdir tb_logs
