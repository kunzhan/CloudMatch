import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from model.semseg.eee_cor_64 import UltraLight_VM_UNet
from dataset.semi import SemiDataset
#from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.deeplabv3plus2fp import DeepLabV3Plus
#from model.semseg.eee import UltraLight_VM_UNet
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', default="/home/zjy/UniMatch-main/UniMatch-main/configs/pascal.yaml", type=str)
parser.add_argument('--labeled-id-path', default="/data/cloud/boime/semi_123/1_32/all_label.txt", type=str) # /semi_123/1_2/all_label.txt
parser.add_argument('--unlabeled-id-path', default="/data/cloud/boime/semi_123/1_32/all_unlabel.txt", type=str) # semi_123/1_2/all_unlabel.txt
parser.add_argument('--save-path', default="/home/zjy/UniMatch-main/UniMatch-main/exp_32/", type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, mode, cfg):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            #pred = model(img).argmax(dim=1)
            res = model(img)
            pred = res.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()



            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            target_meter.update(reduced_target.cpu().numpy())


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    ac = intersection_meter.sum/ (target_meter.sum+ 1e-10)* 100.0
    mIOU = np.mean(iou_class)
    AC =np.mean(ac)
    return mIOU, iou_class, AC