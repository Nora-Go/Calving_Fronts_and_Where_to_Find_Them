import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from .ASPP import ASPP


class UNet(pl.LightningModule):

    def __init__(self, hparams, metric, n_classes):
        super(UNet, self).__init__()
        self.save_hyperparameters(hparams)

        self.n_channels_of_input = 1  # Greyscale
        self.kernel_size = self.hparams.kernel_size
        self.non_linearity = "Leaky_ReLU"
        self.n_layers = 5
        self.features_start = 32
        self.metric = metric
        self.n_classes = n_classes
        self.aspp = True

        self.layers, self.bottleneck = self.make_layer_structure()

    def make_layer_structure(self):
        layers = [DoubleConv(in_channels=self.n_channels_of_input, out_channels=self.features_start,
                             kernel_size=self.kernel_size, non_linearity=self.non_linearity)]

        feats = self.features_start
        for _ in range(self.n_layers - 1):
            layers.append(Down(in_channels=feats, out_channels=feats * 2, kernel_size=self.kernel_size,
                               non_linearity=self.non_linearity))
            feats *= 2

        if self.aspp:
            bottleneck = ASPP(feats, [1, 2, 4, 8], feats)
        else:
            bottleneck = None

        for _ in range(self.n_layers - 1):
            layers.append(Up(in_channels=feats, out_channels=feats // 2, kernel_size=self.kernel_size, non_linearity=self.non_linearity))
            feats //= 2

        layers.append(nn.Conv2d(feats, self.n_classes, kernel_size=1))
        return nn.ModuleList(layers), bottleneck

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.n_layers]:
            xi.append(layer(xi[-1]))

        # Bottleneck layers
        if self.bottleneck is not None:
            xi[-1] = self.bottleneck(xi[-1])

        # Up path
        for i, layer in enumerate(self.layers[self.n_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])

    def adapt_mask(self, y):
        mask_type = torch.float32 if self.n_classes == 1 else torch.long
        y = y.squeeze(1)
        y = y.type(mask_type)
        return y

    def give_prediction_for_batch(self, batch):
        x, y, x_name, y_names = batch

        # Safety check
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
                torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
            print(f"invalid input detected: x {x}, y {y}", file=sys.stderr)

        y_hat = self.forward(x)

        # Safety check
        if torch.any(torch.isnan(y_hat)) or torch.any(torch.isinf(y_hat)):
            print(f"invalid output detected: y_hat {y_hat}", file=sys.stderr)

        return y_hat

    def calc_loss(self, y_hat, y):
        return NotImplemented, NotImplemented

    def make_batch_dictionary(self, loss, metric, name_of_loss):
        return NotImplemented

    def log_metric(self, outputs, train_or_val_or_test):
        pass

    def training_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        assert x.shape[1] == self.n_channels_of_input, \
            f'Network has been defined with {self.n_channels_of_input} input channels, ' \
            f'but loaded images have {x.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
        y_hat = self.give_prediction_for_batch(batch)
        train_loss, metric = self.calc_loss(y_hat, y)

        self.log('train_loss', train_loss)
        return self.make_batch_dictionary(train_loss, metric, "loss")

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Train")

    def validation_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        y_hat = self.give_prediction_for_batch(batch)
        val_loss, metric = self.calc_loss(y_hat, y)
        self.log('val_loss', val_loss)
        return self.make_batch_dictionary(val_loss, metric, "val_loss")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Val")
        self.log('avg_loss_validation', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        y_hat = self.give_prediction_for_batch(batch)
        test_loss, metric = self.calc_loss(y_hat, y)
        self.log('test_loss', test_loss)
        return self.make_batch_dictionary(test_loss, metric, "test_loss")

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr,
                                                      cycle_momentum=False,
                                                      step_size_up=30000)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [scheduler_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        return NotImplemented


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, non_linearity: str):
        super().__init__()
        if non_linearity == "ReLU":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2), nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2), nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, non_linearity: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=kernel_size, non_linearity=non_linearity)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 non_linearity: str):

        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, non_linearity=non_linearity)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
