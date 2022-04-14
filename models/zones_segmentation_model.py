import torch.nn as nn
from models.ParentUNet import UNet
import torchmetrics
import torch
from segmentation_models_pytorch.losses.dice import DiceLoss


class ZonesUNet(UNet):
    def __init__(self, hparams):
        # n_classes = 4 -> Glacier, Rock, Ocean/Ice Melange, NA
        super().__init__(hparams=hparams, metric=torchmetrics.IoU(num_classes=4, reduction="none", absent_score=1.0), n_classes=4)

    def make_batch_dictionary(self, loss, metric, name_of_loss):
        """ Give batch_dictionary corresponding to the number of metrics for zone segmentation """
        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            name_of_loss: loss,
            # info to be used at epoch end
            "IoU": metric[0],
            "IoU NA Area": metric[1],
            "IoU Stone": metric[2],
            "IoU Glacier": metric[3],
            "IoU Ocean and Ice Melange": metric[4]
        }
        return batch_dictionary

    def log_metric(self, outputs, train_or_val_or_test):
        # calculating average metric
        avg_iou = torch.stack([x["IoU"] for x in outputs]).mean()
        avg_iou_na = torch.stack([x["IoU NA Area"] for x in outputs]).mean()
        avg_iou_stone = torch.stack([x["IoU Stone"] for x in outputs]).mean()
        avg_iou_glacier = torch.stack([x["IoU Glacier"] for x in outputs]).mean()
        avg_iou_ocean = torch.stack([x["IoU Ocean and Ice Melange"] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU", avg_iou, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_NA_Area", avg_iou_na, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_Stone", avg_iou_stone, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_Glacier", avg_iou_glacier, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_Ocean_and_Ice_Melange", avg_iou_ocean, self.current_epoch)
        if train_or_val_or_test == "Val":
            self.log('avg_metric_validation', avg_iou)
            self.log('avg_iou_na_validation', avg_iou_na)
            self.log('avg_iou_stone_validation', avg_iou_stone)
            self.log('avg_iou_glacier_validation', avg_iou_glacier)
            self.log('avg_iou_ocean_validation', avg_iou_ocean)

    def class_wise_iou(self, y_hat, y):
        ious = []
        classwise_ious = self.metric(y_hat.argmax(dim=1), y)
        ious.append(torch.mean(classwise_ious))
        for i in classwise_ious:
            ious.append(i)
        return ious

    def calc_loss(self, y_hat, y):
        y = self.adapt_mask(y)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_dice = DiceLoss('multiclass')
        loss = self.hparams.weight_loss * criterion_ce(y_hat, y) + (1 - self.hparams.weight_loss) * criterion_dice(y_hat, y)
        # metric needs to be a list of values (class-wise IoU for zone segmentation)
        # class ids for NA area=0, stone=1, glacier=2, ocean with ice melange=3
        return loss, self.class_wise_iou(y_hat, y)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument('--base_lr', default=4e-5)
        parser.add_argument('--max_lr', default=2e-4)
        parser.add_argument('--batch_size', default=16)
        parser.add_argument('--kernel_size', type=int, default=3)

        # Hyperparameters for augmentation
        parser.add_argument('--bright', default=0.1, type=float)
        parser.add_argument('--wrap', default=0.1, type=float)
        parser.add_argument('--noise', default=0.5, type=float)
        parser.add_argument('--rotate', default=0.5, type=float)
        parser.add_argument('--flip', default=0.3, type=float)

        # Layer arguments
        parser.add_argument('--aspp', default=True, type=lambda x: (str(x).lower() == 'true'))

        # Loss arguments
        parser.add_argument('--weight_loss', default=0.5, type=float)

        return parent_parser
