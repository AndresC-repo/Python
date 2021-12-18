import torch


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    In:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    out:
        list of remaning bboxes
    """

    try:
        # Eliminate bboxes that do not reach the threshold
        bboxes = [box for box in bboxes if box[1] > threshold]
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        remaining_bboxes = []

        while bboxes:
            main_bbox = bboxes.pop(0)
            # bboxes is going to have non-contradictive bboxes
            # - different class bboxes
            # - low Iou (meaning different location)
            bboxes = [
                box
                for box in bboxes
                if box[0] != main_bbox[0]
                or Iou(
                    torch.tensor(main_bbox[2:]),
                    torch.tensor(box[2:]),
                    box_format=box_format,
                )
                < iou_threshold
            ]

            remaining_bboxes.append(main_bbox)
    except TypeError:
        raise Exception("bboxes should be a list")

    return remaining_bboxes


def Iou(boxes_preds, boxes_targets, box_format="midpoint"):
    """
    In:
        boxes_preds: tensor of shape (X1, Y1, X2, Y2)
        boxes_targets: tensor of shape (X1, Y1, X2, Y2)
    Out:
        Intersection over union for all examples as tensor
    """

    # boxes_preds[0].unsqueeze(0) = boxes_preds[..., 0:1]
    if box_format == "corners":
        bb1_x1 = boxes_preds[..., 0:1]
        bb1_y1 = boxes_preds[..., 1:2]
        bb1_x2 = boxes_preds[..., 2:3]
        bb1_y2 = boxes_preds[..., 3:4]
        bb2_x1 = boxes_targets[..., 0:1]
        bb2_y1 = boxes_targets[..., 1:2]
        bb2_x2 = boxes_targets[..., 2:3]
        bb2_y2 = boxes_targets[..., 3:4]

    elif box_format == "midpoint":
        # (x1 -x2)/2
        bb1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        bb1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        bb1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        bb1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        bb2_x1 = boxes_targets[..., 0:1] - boxes_targets[..., 2:3] / 2
        bb2_y1 = boxes_targets[..., 1:2] - boxes_targets[..., 3:4] / 2
        bb2_x2 = boxes_targets[..., 0:1] + boxes_targets[..., 2:3] / 2
        bb2_y2 = boxes_targets[..., 1:2] + boxes_targets[..., 3:4] / 2

    else:
        raise Exception("Define a box format")

    # pdb.set_trace()
    x1 = torch.max(bb1_x1, bb2_x1)
    y1 = torch.max(bb1_y1, bb2_y1)
    x2 = torch.min(bb1_x2, bb2_x2)
    y2 = torch.min(bb1_y2, bb2_y2)

    # clamp to zero in case there is no interection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # get total Union area
    box1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    box2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)
    # 1e-6 to avoid division by zero
    IoU = intersection / (box1_area + box2_area - intersection + 1e-6)
    return IoU


if __name__ == "__main__":
    uno = torch.tensor([1, 0.9, 0, 0, 255, 255])
    dos = torch.tensor([1, 0.7, 0, 0, 200, 200])
    tres = torch.tensor([1, 0.6, 0, 0, 100, 100])

    cuatro = torch.tensor([2, 0.7, 0, 0, 255, 255])
    test = []
    test = [uno, dos, tres, cuatro]
    print(nms(test, 0.5, 0.5))
    print(Iou(uno, dos))
