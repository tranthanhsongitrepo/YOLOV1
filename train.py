import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VOCDataset
from loss import YoloLoss
from model import YoloV1
from utils import get_box, nonmax_supression, mean_average_precision, save_checkpoint, load_checkpoint

seed = 123
torch.manual_seed(seed)

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPTIM = "adam"
GRID_SIZE = 7
NUM_ANCHOR = 2
NUM_CLASS = 20
TRAIN_CSV_PATH = "train.csv"
TEST_CSV_PATH = 'test.csv'
BATCH_SIZE = 16
NUM_WORKERS = 8
WEIGHT_DECAY = 0
EPOCH = 10000
LOAD_MODEL = False
SAVED_MODEL = 'model.pth.tar'
IMAGE_DIR = 'images'
LABEL_PATH = 'labels'
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5
MAP_THRESHOLD_START = 0.5
MAP_THRESHOLD_END = 0.95
MAP_STEP = 0.05
EVAL_AFTER = 100

torch.autograd.set_detect_anomaly(True)


class Compose(object):
    def __init__(self, tfs):
        self.tfs = tfs

    def __call__(self, img, boxes):
        for t in self.tfs:
            img, boxes = t(img), boxes
        return img, boxes


tfs = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train(train_loader, model, optim, loss_fn):
    loop = tqdm.tqdm(train_loader, leave=True)

    for i, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loop.set_postfix(loss=loss.item())


def main():
    model = YoloV1().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(SAVED_MODEL), model, optim)

    train_dataset = VOCDataset(TRAIN_CSV_PATH, IMAGE_DIR, LABEL_PATH, transforms=tfs)
    test_dataset = VOCDataset(TEST_CSV_PATH, IMAGE_DIR, LABEL_PATH, transforms=tfs)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
                             shuffle=True)

    max_mean_ap = 0
    for epoch in range(EPOCH):
        train(train_loader, model, optim, loss_fn)

        if epoch % EVAL_AFTER == 0:
                x, y = get_box(test_loader, model, grid_size=GRID_SIZE, num_class=NUM_CLASS, num_anchors=NUM_ANCHOR, device=DEVICE)
                x = nonmax_supression(x, batch_size=BATCH_SIZE, num_class=NUM_CLASS, iou_threshold=IOU_THRESHOLD,
                                      confidence_threshold=CONFIDENCE_THRESHOLD)
                mean_ap = mean_average_precision(x, y, threshold=MAP_THRESHOLD_START, step=MAP_STEP, end=MAP_THRESHOLD_END, num_class=NUM_CLASS)

                if mean_ap > max_mean_ap:
                  max_mean_ap = mean_ap
                  checkpoint = {
                      "state_dict": model.state_dict(),
                      "optimizer": optim.state_dict(),
                  }
                  save_checkpoint(checkpoint, filename=SAVED_MODEL)

                print("Current mAP: ", mean_ap, "\tBest mAP: ", max_mean_ap)
            
    print("Final mAP: ", mean_average_precision(x, y, threshold=MAP_THRESHOLD_START, step=MAP_STEP, end=MAP_THRESHOLD_END, num_class=NUM_CLASS))

if __name__ == '__main__':
    main()
