import argparse
import argparse
import logging
import os
import pprint

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD,AdamW
from torch.utils.data import DataLoader

import yaml

from dataset.semiself321 import SemiDataset
#from model.semseg.deeplabv3plus import DeepLabV3Plus
# from model.semseg.deeplabv3plus2fp import DeepLabV3Plus
from model.semseg.eee_cor_64 import UltraLight_VM_UNet
from supervised import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import random
import datetime
from einops import rearrange
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # tmux[0]

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', default="/data/xtx/1/configs/pascal.yaml", type=str)
parser.add_argument('--labeled-id-path', default="/data/xtx/semiboime/1_4/all_label.txt", type=str)
parser.add_argument('--unlabeled-id-path', default="/data/xtx/semiboime/1_4/all_unlabel.txt", type=str)
# parser.add_argument('--labeled-id-path', default="/data/xtx/t.txt", type=str)
# parser.add_argument('--unlabeled-id-path', default="/data/xtx/t.txt", type=str)
parser.add_argument('--save-path', default="/data/xtx/1/cloudmacth_4", type=str)
parser.add_argument('--local_rank', default=0, type=int)



def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    model_name = 'Cloudmamba'
    results_file = args.save_path +'/'+ model_name + "results_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # rank, word_size = setup_distributed(port=args.port)
    rank = 0
    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    init_seeds(0, False)

    # model = DeepLabV3Plus(cfg)
    model = UltraLight_VM_UNet()
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.parameters(), 'lr': cfg['lr']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    model.cuda()

  
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    mse_loss = nn.MSELoss().cuda()
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    # trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=4, drop_last=True , shuffle=True)
    # trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=4, drop_last=True, shuffle=True)#sampler=None
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=False, sampler=None)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    #thresh_controller = ThreshController(nclass=2, momentum=0.999, thresh_init=cfg['thresh_init'])
    
    p_model = (torch.ones(cfg['nclass']) / cfg['nclass']).cuda()
    label_hist = (torch.ones(cfg['nclass']) / cfg['nclass']).cuda() 
    time_p = p_model.mean()


    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.6f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp, total_loss_u_s_mse = 0.0, 0.0, 0.0, 0.0,0.0

        total_mask_ratio = 0.0

        # trainloader_l.sampler.set_epoch(epoch)
        # trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        if rank == 0:
            tbar = tqdm(total=len(trainloader_l))

        for i, ((img_x, mask_x),
                (img_u1_w1, img_u1_s1, img_u1_s2, img_u1_w2, cutmix_box1, cutmix_box2),
                (img_u2_w1, img_u2_s1, img_u2_s2, img_u2_w2, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()

            img_u1_w1 = img_u1_w1.cuda()
            img_u1_s2 = img_u1_s2.cuda()
            img_u2_s2=img_u2_s2.cuda()
            img_u2_w1 = img_u2_w1.cuda()
            img_u1_w2 =img_u1_w2.cuda()
            img_u2_w2 =img_u2_w2.cuda()
            img_u1_s1 = img_u1_s1.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            cutmix_box2 = cutmix_box2.cuda()
            img_u1_w_cut = img_u1_w1.clone().cuda()
            img_u2_w_cut = img_u2_w1.clone().cuda()
            img_u1_s_cut = img_u1_s1.clone().cuda()
            img_u2_s_cut = img_u2_s1.clone().cuda()

            with torch.no_grad():
                model.eval()
                pred_u1_w2_f = model(img_u1_w2, need_fp=False).detach()
                conf_u1_w2_f = pred_u1_w2_f.softmax(dim=1).max(dim=1)[0]
                mask_u1_w2_f = pred_u1_w2_f.argmax(dim=1)

                pred_u2_w2_f = model(img_u2_w2, need_fp=False).detach()
                conf_u2_w2_f = pred_u2_w2_f.softmax(dim=1).max(dim=1)[0]
                mask_u2_w2_f = pred_u2_w2_f.argmax(dim=1)

            
            img_u1_w_cut[cutmix_box1.unsqueeze(1).expand(img_u1_w_cut.shape) == 1] = img_u1_w2[cutmix_box1.unsqueeze(1).expand(img_u1_w_cut.shape) == 1]
            img_u1_s_cut[cutmix_box1.unsqueeze(1).expand(img_u1_w_cut.shape) == 1] = img_u1_s2[cutmix_box1.unsqueeze(1).expand(img_u1_w_cut.shape) == 1]
            
            img_u2_w_cut[cutmix_box2.unsqueeze(1).expand(img_u2_w_cut.shape) == 1] = img_u2_w2[cutmix_box2.unsqueeze(1).expand(img_u2_w_cut.shape) == 1]
            img_u2_s_cut[cutmix_box2.unsqueeze(1).expand(img_u2_w_cut.shape) == 1] = img_u2_s2[cutmix_box2.unsqueeze(1).expand(img_u2_w_cut.shape) == 1]
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u1_w1.shape[0]


            preds = model(torch.cat((img_x, img_u1_w1,img_u2_w1)), need_fp=False)# , use_corr=False
            pred_x, pred_u1_w1, pred_u2_w1 = preds.split([num_ulb, num_ulb, num_ulb])
            res_s = model(torch.cat((img_u1_s_cut, img_u2_s_cut,img_u1_w_cut,img_u2_w_cut)), need_fp=False)# , use_corr=False
            pred_u1_s1, pred_u2_s2,pred_u1_w2,pred_u2_w2 = res_s.split([num_lb, num_ulb,num_lb, num_ulb])


            pred_u1_w1 = pred_u1_w1.detach()
            conf_u1_w = pred_u1_w1.softmax(dim=1).max(dim=1)[0]
            mask_u1_w = pred_u1_w1.argmax(dim=1)
            
            pred_u2_w1 = pred_u2_w1.detach()
            conf_u2_w = pred_u2_w1.softmax(dim=1).max(dim=1)[0]
            mask_u2_w = pred_u2_w1.argmax(dim=1)

            mask_u1_w_cutmixed1, conf_u1_w_cutmixed1,pred_u1_w_cutmixed1, mask_u2_w_cutmixed2 , conf_u2_w_cutmixed2,pred_u2_w_cutmixed1= \
                mask_u1_w.clone(), conf_u1_w.clone(), pred_u1_w1.clone(),mask_u2_w.clone(), conf_u2_w.clone(), pred_u2_w1.clone()
            cutmix_box1_expanded = cutmix_box1.unsqueeze(1)
            cutmix_box1_broadcasted = cutmix_box1_expanded.repeat(1, 2, 1, 1) 
            cutmix_box2_expanded = cutmix_box2.unsqueeze(1) 
            cutmix_box2_broadcasted = cutmix_box2_expanded.repeat(1, 2, 1, 1) 
            mask_u1_w_cutmixed1[cutmix_box1 == 1] = mask_u1_w2_f[cutmix_box1 == 1]
            conf_u1_w_cutmixed1[cutmix_box1 == 1] = conf_u1_w2_f[cutmix_box1 == 1]
            pred_u1_w_cutmixed1[cutmix_box1_broadcasted == 1] = pred_u1_w2_f[cutmix_box1_broadcasted == 1]

            mask_u2_w_cutmixed2[cutmix_box2 == 1] = mask_u2_w2_f[cutmix_box2 == 1]
            conf_u2_w_cutmixed2[cutmix_box2 == 1] = conf_u2_w2_f[cutmix_box2 == 1]
            pred_u2_w_cutmixed1[cutmix_box2_broadcasted == 1] = pred_u2_w2_f[cutmix_box2_broadcasted == 1]

            time_p, p_model, label_hist = cal_time_p_and_p_model(pred_u1_w1, time_p, p_model, label_hist)
            pseudo_label = torch.mean(pred_u1_w1, dim=(2,3))
            pseudo_label = torch.softmax(pseudo_label, dim=1)
            
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            p_cutoff = time_p
            p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]
            threshold = torch.mean(p_cutoff * p_model_cutoff[max_idx])
            conf_fliter_u_w = (conf_u1_w_cutmixed1 >= threshold)
            conf_fliter_u_w2 = (conf_u2_w_cutmixed2 >= threshold)
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u1_s1, mask_u1_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * conf_fliter_u_w
            loss_u_s1 = torch.sum(loss_u_s1) / (4*384*384)

            loss_u_s2 = criterion_u(pred_u2_s2, mask_u2_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * conf_fliter_u_w2
            loss_u_s2 = torch.sum(loss_u_s2) / (4*384*384)
            
            
            loss_u_w1 = criterion_u(pred_u1_w2, mask_u1_w_cutmixed1)
            loss_u_w1 = loss_u_w1 * conf_fliter_u_w
            loss_u_w1 = torch.sum(loss_u_w1) / (4*384*384)
            
            loss_u_w2 = criterion_u(pred_u1_w2, mask_u2_w_cutmixed2)
            loss_u_w2 = loss_u_w2 * conf_fliter_u_w2
            loss_u_w2 = torch.sum(loss_u_w2) / (4*384*384)
            
            loss_u_w = loss_u_w1+loss_u_w2
            pred_u_s1_norm = rearrange(pred_u1_s1, 'n c h w -> n c (h w)')
            pred_u_s1_norm = (pred_u_s1_norm - torch.mean(pred_u_s1_norm, dim=2, keepdim=True)) / torch.std(pred_u_s1_norm, dim=2, keepdim=True)

            pred_u_s2_norm = rearrange(pred_u1_w2, 'n c h w -> n c (h w)')
            pred_u_s2_norm = (pred_u_s2_norm - torch.mean(pred_u_s2_norm, dim=2, keepdim=True)) / torch.std(pred_u_s2_norm, dim=2, keepdim=True)
              
            loss_u_s_mse = mse_loss(pred_u_s1_norm, pred_u_s2_norm)
            
            pred_u_s3_norm = rearrange(pred_u2_s2, 'n c h w -> n c (h w)')
            pred_u_s3_norm = (pred_u_s3_norm - torch.mean(pred_u_s3_norm, dim=2, keepdim=True)) / torch.std(pred_u_s3_norm, dim=2, keepdim=True)

            pred_u_s4_norm = rearrange(pred_u2_w2, 'n c h w -> n c (h w)')
            pred_u_s4_norm = (pred_u_s4_norm - torch.mean(pred_u_s4_norm, dim=2, keepdim=True)) / torch.std(pred_u_s4_norm, dim=2, keepdim=True)
              
            loss_u_s_mse2 = mse_loss(pred_u_s3_norm, pred_u_s4_norm)
            loss_u_s_mse2 = loss_u_s_mse2
            loss = loss_x + (0.5*loss_u_s1 + 0.5*loss_u_s2 + 0.5*loss_u_w+ 0.5*loss_u_s_mse + 0.5*loss_u_s_mse2)/2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += (loss_u_s1.item()+loss_u_s2.item())/2.0
            total_loss_w_fp += loss_u_s_mse2.item()
            total_loss_u_s_mse += loss_u_s_mse.item()
            total_mask_ratio += ((conf_u1_w >=threshold)).sum().item() / (4*384*384)
                                

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                tbar.set_description(' Total loss: {:.3f}, Loss x: {:.3f} '
                                     'Loss s: {:.3f}, Loss w_fp: {:.3f}, loss s_mse: {:.3f}, Mask: {:.3f}'.format(total_loss / (i + 1), total_loss_x / (i + 1), total_loss_s / (i + 1),
                    total_loss_w_fp / (i + 1),total_loss_u_s_mse / (i + 1), total_mask_ratio / (i + 1)))
                tbar.update(1)

        if rank == 0:
            tbar.close()

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        mIOU,class_IOU,AC = evaluate(model, valloader, eval_mode, cfg)
        # torch.distributed.barrier()
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {total_loss / (i + 1):.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"val_mIOU:{mIOU} \n"\
                         f"val_class_IOU:{class_IOU}\n"\
            # f.write(train_info + val_info + "\n\n")
            f.write(train_info+ "\n\n")
        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.4f} \n'.format(eval_mode, mIOU))
            logger.info('***** Evalu ***** >>>> ac: {:.4f} \n'.format(AC))

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            # torch.save(model.module.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU)))
            torch.save(model.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU)))
        # torch.distributed.barrier()
        
@torch.no_grad()
def cal_time_p_and_p_model(logits_x_ulb_w, time_p, p_model, label_hist):
    
    logits_x_ulb_w = torch.mean(logits_x_ulb_w, dim=(2,3))
    prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
    max_probs, max_idx = torch.max(prob_w, dim=-1)
    
    if time_p is None:
        time_p = max_probs.mean()
    else:
        time_p = time_p * 0.999 +  max_probs.mean() * 0.001
    if p_model is None:
        p_model = torch.mean(prob_w, dim=1)
    else:
        p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
    if label_hist is None:
        label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
        label_hist = label_hist / label_hist.sum()
    else:
        hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 

        label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
    return time_p,p_model,label_hist


if __name__ == '__main__':
    main()
