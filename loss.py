import torch
from utils import IOU
from torch import nn


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.lambda_coord = 5
        self.lambda_noobj = .5
        self.S = 7
        self.B = 2
        self.C = 20
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, x, y):
        x = x.reshape(-1, self.S, self.S, self.C + self.B * 5)

        ious = torch.zeros((self.B, x.shape[0], self.S, self.S))
        for b in range(self.B):
            iou = IOU(x[..., b * 5 + self.C + 1:(b + 1) * 5 + self.C], y[..., self.C + 1:])
            ious[b, :, :, :] = iou.squeeze(-1)

        best_ious, best_box = torch.max(ious, dim=0)

        one_i = y[..., self.C].unsqueeze(3)

        box_loss = 0
        obj_loss = 0
        noobj_loss = 0
        class_loss = 0
        
        for i in range(self.B):
            one_j = torch.where(best_box == i, 1, 0).to('cuda').unsqueeze(-1)
            box_x = one_j * x[..., i * 5 + self.C + 1:(i + 1) * 5 + self.C]

            box_loss += box_x
            
            obj_x = one_j * x[..., self.C + i * 5:self.C + i * 5 + 1]
            obj_loss += obj_x

            noobj_x = (1 - one_i) * (1 - one_j) * x[..., self.C + i * 5:self.C + i * 5 + 1]
            noobj_loss += noobj_x
            # print(noobj_x.shape, noobj_y.shape)

        class_x = one_i * x[..., :self.C]
        class_y = one_i * y[..., :self.C]

        box_loss = one_i * box_loss
        box_loss[..., 2:4] = torch.sign(box_loss[..., 2:4].clone()) * torch.sqrt(torch.abs(box_loss[..., 2:4].clone()) + 1e-6)
        box_y = one_i * y[..., self.C + 1:]
        box_y[..., 2:4] = torch.sign(box_y[..., 2:4].clone()) * torch.sqrt(torch.abs(box_y[..., 2:4].clone()) + 1e-6)

        best_ious = best_ious.unsqueeze(-1)
        obj_y = one_i * best_ious.to('cuda')
        noobj_y = (1 - one_i) * best_ious.to('cuda')

        box_loss = self.mse(box_loss.flatten(end_dim=-2), box_y.flatten(end_dim=-2))
        obj_loss = self.mse(obj_loss.flatten(), obj_y.flatten())
        noobj_loss = self.mse(noobj_loss.flatten(start_dim=1), noobj_y.flatten(start_dim=1))
        class_loss = self.mse(class_x.flatten(end_dim=-2), class_y.flatten(end_dim=-2))
                  
        return self.lambda_coord * box_loss + obj_loss + self.lambda_noobj * noobj_loss + class_loss
