from collections import Counter
from typing import re

import torch


def IOU(x, y):
    x_x1, x_x2, x_y1, x_y2 = x[..., 0:1] - x[..., 2:3] / 2, \
                            x[..., 0:1] + x[..., 2:3] / 2, \
                            x[..., 1:2] - x[..., 3:4] / 2, \
                            x[..., 1:2] + x[..., 3:4] / 2
    y_x1, y_x2, y_y1, y_y2 = y[..., 0:1] - y[..., 2:3] / 2, \
                            y[..., 0:1] + y[..., 2:3] / 2, \
                            y[..., 1:2] - y[..., 3:4] / 2, \
                            y[..., 1:2] + y[..., 3:4] / 2

    x1 = torch.max(x_x1, y_x1)
    x2 = torch.min(x_x2, y_x2)
    y1 = torch.max(x_y1, y_y1)
    y2 = torch.min(x_y2, y_y2)
    box1_area = torch.abs(x_x1 - x_x2) * torch.abs(x_y1 - x_y2)
    box2_area = torch.abs(y_x1 - y_x2) * torch.abs(y_y1 - y_y2)
    interception = torch.abs((y2 - y1).clamp(0) * (x2 - x1).clamp(0))
    return interception / (box1_area + box2_area - interception + 1e-6)


def mean_average_precision(x, y, threshold=0.5, step=0.05, end=0.95, num_class=20):
    average_precision = []
    for thresh in torch.arange(threshold, end, step):
        average_precision_per_class = []
        for c in range(num_class):
            x_boxes = x[x[..., 1] == c]
            y_boxes = y[y[..., 1] == c]
            
            tp = torch.zeros((x_boxes.shape[0]))
            fp = torch.zeros((x_boxes.shape[0]))
            
            if y_boxes.shape[0] == 0:
                continue

            # Sort the tensor by confidence
            x_boxes = x_boxes[x_boxes[..., 2].argsort(descending=True)]
            total_y = y_boxes.shape[0]
          
            # Returns a dictionary of {0:n, ... m:n} where m is the number of examples in y and n is the number of boxes in that example
            taken = Counter([int(gt[0]) for gt in y_boxes])

            for key in taken.keys():
                taken[key] = torch.zeros(taken[key])

            for i, x_box in enumerate(x_boxes):
                y_boxes_same_img = [box for box in y_boxes if box[0] == x_box[0]]
                best_iou = 0
                best_idx = -1
                for idx, y_box_same_img in enumerate(y_boxes_same_img):
                    # print(y_box_same_img[3:])
                    iou = IOU(x_box[3:], y_box_same_img[3:])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou > thresh and taken[int(x_box[0])][best_idx] == 0:
                    tp[i] = 1
                    taken[int(x_box[0])][best_idx] = 1
                else:
                    fp[i] = 1

            # if tp[tp == 1].shape[0] != 0:
            #   print(tp[tp == 1])
            tp_cumsum = tp.cumsum(dim=0)
            fp_cumsum = fp.cumsum(dim=0)

            recall = tp_cumsum / (total_y + 1e-6)
            precision = tp_cumsum / (fp_cumsum + tp_cumsum + 1e-6)

            precision = torch.cat((torch.tensor([1]), precision))
            recall = torch.cat((torch.tensor([0]), recall))
            average_precision_per_class.append(torch.trapz(precision, recall))

        average_precision_per_class = torch.FloatTensor(average_precision_per_class).to('cuda')
        average_precision.append(average_precision_per_class.mean())

    return torch.FloatTensor(average_precision).to('cuda').mean()


def nonmax_supression(x, batch_size=16, num_class=20, iou_threshold=0.5, confidence_threshold=0.5):
    res = []
    all_boxes = x[x[:, 2] > confidence_threshold]
    all_boxes = all_boxes[all_boxes[:, 2].argsort(descending=True)]

    for i in range(batch_size):
      boxes = all_boxes[all_boxes[:, 0] == i]
      while boxes.shape[0] != 0:
        max_box = boxes[0]

        boxes = boxes[1:]
        
        ious = IOU(max_box[3:].unsqueeze(0).repeat(boxes.shape[0], 1), boxes[..., 3:])
        boxes = boxes[boxes[:, 1] != max_box[1].logical_or((ious < iou_threshold).squeeze(-1))]
        
        res.append(max_box)

    if len(res) > 0:
      return torch.stack(res)
    else:
      return torch.empty((0, 7))


def cell2box(x, num_class=20, device='cuda'):
    x = x.to(device)
    idxs = torch.arange(x.shape[0]).unsqueeze(-1).repeat_interleave(49, dim=1).reshape((-1, x.shape[1], x.shape[1])).unsqueeze(-1).to(device)
    classes = x[..., :num_class].argmax(dim=-1).unsqueeze(-1).to(device)
    box = x[..., num_class:]
    grid_idxs = torch.arange(7).repeat(x.shape[0], x.shape[1], 1).unsqueeze(-1).to(device)
    
    box[..., 1:2] = (box[..., 1:2] + grid_idxs) / x.shape[1]
    box[..., 2:3] = (box[..., 2:3] + grid_idxs.permute(0, 2, 1, 3).to(device)) / x.shape[1]
    box[..., 3:4] = box[..., 3:4] / x.shape[1]
    box[..., 4:] = box[..., 4:] / x.shape[1]

    return torch.cat((idxs, classes, box),  dim=-1).reshape((-1, 7))


def get_box(test_loader, model, num_class=20, grid_size=7, confidence_threshold=0.5, iou_threshold=0.5, num_anchors=2, device="cuda"):
    model.eval()
    all_x = torch.empty((0, 7))
    all_y = torch.empty((0, 7))
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)
        pred = pred.reshape(-1, grid_size, grid_size, num_class + num_anchors * 5)

        boxes = torch.zeros((num_anchors, x.shape[0], pred.shape[1], pred.shape[2], num_class + 5))
        for anchor in range(num_anchors):
            boxes[anchor, :, :, :, num_class:num_class + 5] = pred[..., num_class + anchor * 5:num_class + (anchor + 1) * 5]
            boxes[anchor, :, :, :, :num_class] = pred[..., :num_class]

        best_box, _ = boxes.max(dim=0)

        best_box = cell2box(best_box, num_class)
        y = cell2box(y, num_class)

        y = y[y[..., 2] > confidence_threshold]
        model.train()
        # print(y[..., 2:3])
        all_x = torch.cat((all_x, best_box))
        all_y = torch.cat((all_y, y))
    return all_x, all_y


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])