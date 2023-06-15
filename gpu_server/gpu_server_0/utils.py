
import sys
import os
import cv2
import torchvision.transforms as T
import torch.nn as nn
from medpy import metric
import torch
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        assert inputs.size() == target.size(
        ), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        sigmoid = nn.Sigmoid()
        inputs = sigmoid(inputs)
        loss = self._dice_loss(inputs, target)

        return loss

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.cuda(), label.cuda()
    sigmoid = nn.Sigmoid()
    net.eval()
    with torch.no_grad():
        first_output, second_output = net(image)
        outputs = torch.round(sigmoid(first_output))
        outputs2 = torch.round(sigmoid(second_output))

    outputs, outputs2, label = outputs.cpu().detach().numpy(
    ), outputs2.cpu().detach().numpy(), label.cpu().detach().numpy()

    if test_save_path is not None:
        for i in range(1):
            visualization(outputs2[0][i], label[0][i],
                          case, test_save_path, i, 2)


def visualization(pred, label, path, s_path, n_class, n):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))  # 224 224 3

    TP = np.zeros(pred.shape)

    for i in range(0, pred.shape[0]):
        for j in range(0, pred.shape[0]):
            if (pred[i, j] == 1) and (label[i, j] != 0):
                TP[i, j] = 1
            elif (label[i, j] == 0) and (pred[i, j] == 1):
                TP[i, j] = 2  # FP
            elif (label[i, j] != 0) and (pred[i, j] == 0):
                TP[i, j] = 3  # FN

    # Prediction only
    for i in range(0, pred.shape[0]):
        for j in range(0, pred.shape[0]):
            if TP[i, j] == 1:  # TP
                image[i, j, 0] = 0
                image[i, j, 1] = 255
                image[i, j, 2] = 0
            elif TP[i, j] == 2:  # FP
                # print("--FP")
                image[i, j, 0] = 0
                image[i, j, 1] = 255
                image[i, j, 2] = 0

    name = ['bolus', 'cervical spine', 'hyoid bone',
            'mandible', 'soft tissue', 'vocal folds']
    path = path.split("/")
    path = path[len(path)-1]
    cv2.imwrite(os.path.join(s_path, path), image)