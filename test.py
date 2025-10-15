import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch  
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD,AdamW
from torch.utils.data import DataLoader
import yaml
from dataset.transform import *
from dataset.semi import SemiDataset
from model.semseg.eee import UltraLight_VM_UNet
# from model.semseg.cm import UltraLight_VM_UNet
from supervised import evaluate
from model.semseg.mamba import Network
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
import random
from einops import rearrange
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # "0, 1, 2, 3"

import warnings
import numpy as np
from PIL import Image 
warnings.filterwarnings("ignore")



def main():
    rot = '/data/xtx/othertest'
    name = 'tcloud'
    sys.path.append(rot+ '/'+name)

    resume_model = os.path.join("1/rice/resnet101_82.415.pth")
    outputs = os.path.join(rot+ '/'+name)

    if not os.path.exists(outputs):
        os.makedirs(outputs)

    # init_seeds(0, False)
    gpu_ids = [0]
    torch.cuda.empty_cache()
    


    print('#----------Prepareing Models----------#') 
    model = UltraLight_VM_UNet()
    # model = Network(num_classes=2, criterion=None,
    #                 pretrained_model=None,
    #                 norm_layer=None)
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    model.eval()

    checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
    # checkpoint=checkpoint['model']
    model.module.load_state_dict(checkpoint)
    print('#----------Testing----------#')

    to_test = {'test':'/data/xtx/38test'}
    # val_path = '/data/xtx/t.txt'
    val_path = '/data/xtx/T-Cloud/train/train.txt'
    # val_path = '/data/xtx/large_images.txt'
    

    with torch.no_grad():
        TN = 0
        FP = 0
        FN = 0
        TP = 0
        for name, root in to_test.items():

            # 获取图片名称list,txt
            with open(val_path, 'r') as f:
                ids = f.read().splitlines()
            img_list = ids
            
            # 获取图片名称list,遍历文件夹
            # img_list = [os.path.splitext(f)[0] for f in os.listdir(root+'/mask') if f.endswith('.png')]
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))

                # img = Image.open(os.path.join(root+'/38img/', img_name)).convert('RGB')
                # mask = Image.fromarray(np.array(Image.open(os.path.join(root+'/38mask/', img_name )))/255)
                img = Image.open(os.path.join('/data/xtx/T-Cloud/train/cloud/', img_name)).convert('RGB')
                # mask = Image.fromarray(np.array(Image.open(os.path.join('/data/xtx/T-Cloud/train/reference/', img_name)))/255)
                img_path = os.path.join('/data/xtx/T-Cloud/train/reference/', img_name)
                mask = Image.open(img_path).convert('L')
                mask = np.array(mask).astype(np.float32) / 255.0  # 归一化到 [0,1]，使用 float32
                # # 如果你需要转回 PIL 图像：
                # mask = Image.fromarray((arr * 255).astype(np.uint8))
                img, mask = normalize(img, mask)
                img= img.unsqueeze(0)
                mask= mask.unsqueeze(0)
                #pred = model(img).argmax(dim=1)
                # res = model(img,step = 1)
                res = model(img)
                probs = torch.softmax(res, dim=1)
                pred = res.argmax(dim=1)

                # intersection, union, target = \
                #     intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), 2, 255)

                # reduced_intersection = torch.from_numpy(intersection).cuda()
                # reduced_union = torch.from_numpy(union).cuda()
                # reduced_target = torch.from_numpy(target).cuda()



                # intersection_meter.update(reduced_intersection.cpu().numpy())
                # union_meter.update(reduced_union.cpu().numpy())
                # target_meter.update(reduced_target.cpu().numpy())


                # 如果是 float，转 uint8
                # pred = (pred * 255).astype(np.uint8)
                class_id = 1
                pred_prob = probs[0, class_id]  # shape: [H, W], dtype: float32, 范围 [0, 1]

                # 3. 转 numpy
                pred_np = pred_prob.cpu().numpy()  # shape: [H, W], dtype: float32

                # 4. 转 PIL Image（PIL 要求 uint8，所以要缩放到 [0,255]）
                pred_uint8 = (pred_np * 255).astype(np.uint8)

                img = Image.fromarray(pred_uint8, mode='L')  
                img.save(outputs + '/' + img_name)


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(iou_class) * 100.0
    ac=intersection_meter.sum/ (target_meter.sum+ 1e-10)
    AC = np.mean(ac) * 100.0
    print(mIOU,AC)



if __name__ == '__main__':
    main()