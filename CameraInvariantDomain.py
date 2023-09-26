import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
from torch.optim.lr_scheduler import StepLR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from InternimageMask2Former import Itern_Mask2Former
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss
import torch
import torch.nn as nn
from mmseg.models.builder import LOSSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()

    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights = bin_label_weights * valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False,
                         **kwargs):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
            Note: In bce loss, label < 0 is invalid.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.size(1) == 1:
        # For binary class segmentation, the shape of pred is
        # [N, 1, H, W] and that of label is [N, H, W].
        assert label.max() <= 1, \
            'For pred with shape [N, 1, H, W], its label must have at ' \
            'most 2 classes'
        pred = pred.squeeze()
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        # `weight` returned from `_expand_onehot_labels`
        # has been treated for valid (non-ignore) pixels
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.shape, ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    # average loss over non-ignored and valid elements
    if reduction == 'mean' and avg_factor is None and avg_non_ignore:
        avg_factor = valid_mask.sum().item()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]





def compute_miou(y_true, y_pred):
    intersection = torch.logical_and(y_true, y_pred)
    union = torch.logical_or(y_true, y_pred)
    miou = torch.sum(intersection) / torch.sum(union)
    return miou


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def apply_fisheye_distortion(image_path, mask=False):
    # 이미지 불러오기
    if mask == True:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image[image == 255] = 12
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 크기 가져오기
    height, width = image.shape[:2]

    # 카메라 매트릭스 생성
    focal_length = width / 4
    center_x = width / 2
    center_y = height / 2
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)

    # 왜곡 계수 생성
    dist_coeffs = np.array([0, 0.2, 0, 0], dtype=np.float32)

    # 왜곡 보정
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    if mask == True:
        undistorted_image = undistorted_image[40:height, 340:1740]
    else:
        undistorted_image = undistorted_image[40:height, 340:1740, :]
    return (undistorted_image)


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.infer:
            if self.transform:
                img_path = self.data.iloc[idx, 1]
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transform(image=image)['image']
            return image

        img_path = self.data.iloc[idx, 1]
        mask_path = self.data.iloc[idx, 2]

        image = apply_fisheye_distortion(img_path)
        mask = apply_fisheye_distortion(mask_path, mask=True)

        # 배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


submiting_transform = A.Compose(
    [
        A.Resize(960, 540),
        A.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ToTensorV2()
    ]
)




training_transform = A.Compose(
    [
        A.Resize(960, 540),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ToTensorV2()
    ]
)

if __name__ == '__main__':
    BATCH = 2
    model = Itern_Mask2Former().cuda()

    dataset = CustomDataset(csv_file='./train_source.csv', transform=training_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=1)
    criterion =nn.CrossEntropyLoss()
    val_dataset = CustomDataset(csv_file='./val_source.csv', transform=training_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True, num_workers=1)

    model.load_state_dict(torch.load('./latest.pt'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    biggest_miou = 0.8869878721339508

    for epoch in range(200):  # 20 에폭 동안 학습합니다.
        model.train()
        epoch_miou = 0
        train_loss = 0
        count = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            count += 1

            if (count % 100) == 99:
                time.sleep(9)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        print(f'Epoch {epoch + 1}, Train_Loss: {train_loss / len(dataloader)}')
        model.eval()
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader):
                images = images.float().to(device)
                masks = masks.long().to(device)
                count += 1
                if (count % 100) == 99:
                    time.sleep(9)
                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)
                miou = compute_miou(outputs, masks.squeeze(1))
                epoch_miou += miou.item()


        print(f'Epoch {epoch + 1}, MIOU: {epoch_miou / len(val_dataloader)}')
        if biggest_miou < epoch_miou / len(val_dataloader):
            print("changed")
            biggest_miou = epoch_miou / len(val_dataloader)
            torch.save(model.state_dict(), './best_' + str(biggest_miou) + '.pt')
        else :
            torch.save(model.state_dict(), './latest.pt')

    test_dataset = CustomDataset(csv_file='./test.csv', transform=submiting_transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=1)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()

            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred)  # 이미지로 변환
                pred = pred.resize((960, 540), Image.NEAREST)  # 960 x 540 사이즈로 변환
                pred = np.array(pred)  # 다시 수치로 변환
                # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0:  # 마스크가 존재하는 경우 encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else:  # 마스크가 존재하지 않는 경우 -1
                        result.append(-1)
    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./baseline_submit.csv', index=False)

"""
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-8)
    for epoch in range(5):  # 20 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        count =0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            count+=1
            if(count%100) ==99:
                time.sleep(9)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}')
    torch.save(model.state_dict(),'./Iterated.pt')
"""
