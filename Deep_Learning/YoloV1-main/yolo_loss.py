import torch
import torch.nn as nn
from IoU import Iou


class Yolo_loss(nn.Module):
    # Loss with values as in original paper
    def __init__(self, S=7, B=2, C=20):
        super(Yolo_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        Split size (S) = 7
        Number of boxes created (B) = 2
        Classes | VOC (C) = 20
        """
        self.S = S
        self.B = B
        self.C = C

        # lambda_noobj loss for no Object present
        # lambda_coord related to coordinates of the boxes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):

        # Calculate IoU for the two predicted bounding boxes with target bbox
        # [0-19 are Classes, 20 ProbScore, 21-25 coordinates]
        iou_b1 = Iou(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = Iou(predictions[..., 26:30], target[..., 21:25])

        # chosen_box: box with highest IoU out of the two prediction
        _, chosen_box = torch.max(
            torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0), dim=0)
        Iobj_i = target[..., 20].unsqueeze(3)  # if the obecejt exists

        # --------------------------------- BOX COORDINATES LOSS ---------------------------------

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously =estox
        box_predictions = Iobj_i * (
            (
                chosen_box * predictions[..., 26:30] + (1 - chosen_box) * predictions[..., 21:25]  # noqa: E501
            )
        )

        box_targets = Iobj_i * target[..., 21:25]

        # Sqrt of h,w
        box_predictions[..., 2:4] = torch.sqrt(box_predictions[..., 2:4])
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # LOSS MSE coordinates best box vs target
        coord_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        coord_loss = self.lambda_coord * coord_loss

        # --------------------------------- OBJECT LOSS ---------------------------------

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (chosen_box * predictions[..., 25:26] + (1 - chosen_box) * predictions[..., 20:21])  # noqa: E501

        # MSE best box prediction confidence vs target label (if tarjet exists)
        object_loss = self.mse(
            torch.flatten(Iobj_i * pred_box),
            torch.flatten(Iobj_i * target[..., 20:21]),
        )

        # This will give loss if no object is present Iobj_i = 0
        no_object_loss = self.mse(
            torch.flatten((1 - Iobj_i) * predictions[..., 20:21], start_dim=1),  # noqa: E501
            torch.flatten((1 - Iobj_i) * target[..., 20:21], start_dim=1),  # noqa: E501
        )

        no_object_loss += self.mse(
            torch.flatten((1 - Iobj_i) * predictions[..., 25:26], start_dim=1),  # noqa: E501
            torch.flatten((1 - Iobj_i) * target[..., 20:21], start_dim=1)
        )
        no_object_loss = self.lambda_noobj * no_object_loss
        # --------------------------------- CLASS LOSS ---------------------------------
        # Class loss first 20 digits of preds and targets
        class_loss = self.mse(
            torch.flatten(Iobj_i * predictions[..., :20], end_dim=-2,),
            torch.flatten(Iobj_i * target[..., :20], end_dim=-2,),
        )

        # --------------------------------- TOTAL LOSS ---------------------------------
        total_loss = (coord_loss + object_loss + no_object_loss + class_loss)

        return total_loss


if __name__ == "__main__":
    # ------------------------
    # Test
    # ------------------------
    criterion = Yolo_loss()

    predictions = torch.rand([1, 7, 7, 30])
    labels = torch.rand([1, 7, 7, 30])
    print(criterion(predictions, labels))
