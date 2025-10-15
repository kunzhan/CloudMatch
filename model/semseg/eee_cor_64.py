import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
import sys
sys.path.append('/data/CorrMatch-main/model')
import math
from einops import rearrange
# from mamba_ssm import Mamba
from .mamba_space import Mamba
from .block import DANetHead

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.norm1 = nn.LayerNorm(input_dim*2)
        self.mamba = Mamba(
                d_model=input_dim//2, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2"
        )
        self.proj = nn.Linear(input_dim*2, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        x1_copy = x1
        x1 = torch.cat([x1_copy,x1_copy],dim=2)
        x2_copy = x2
        x2 = torch.cat([x2_copy,x2_copy],dim=2)
        x3_copy = x3
        x3 = torch.cat([x3_copy,x3_copy],dim=2)
        x4_copy = x4
        x4 = torch.cat([x4_copy,x4_copy],dim=2)
        

        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
       
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
       

        x_mamba = self.norm1(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1), 
                         self.avgpool(t2), 
                         self.avgpool(t3), 
                         self.avgpool(t4), 
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)
            
        return att1, att2, att3, att4, att5
    
    
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())
    
    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]

    
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3, t4, t5):
        # 原始
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        # 
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5
        # satt
        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_
class DA_Att_Bridge1(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.datt1 = DANetHead(8,8)
        self.datt2 = DANetHead(16,16)
        self.datt3 = DANetHead(24,24)
        self.datt4 = DANetHead(32,32)
        self.datt5 = DANetHead(48,48)
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3, t4, t5):
        # 原始
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        # 
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5
        # satt
        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5
        t1 = t1+r1_
        t2 = t2+r2_


        t3, t4, t5 = self.datt3(r3),self.datt4(r4),self.datt5(r5)
        # catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        # t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5
   
        return t1, t2, t3, t4, t5
class DA_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.datt1 = DANetHead(8,8)
        self.datt2 = DANetHead(16,16)
        self.datt3 = DANetHead(24,24)
        self.datt4 = DANetHead(32,32)
        self.datt5 = DANetHead(48,48)
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3, t4, t5):
        # 原始
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        # 
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5
        # satt
        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        t1, t2, t3, t4, t5 = self.datt1(t1),self.datt2(t2),self.datt3(t3),self.datt4(t4),self.datt5(t5)

        return t1, t2, t3, t4, t5

class UltraLight_VM_UNet(nn.Module):
    
    def __init__(self, num_classes=2, input_channels=3, c_list=[8,16,24,32,48,64],
                split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            PVMLayer(c_list[0], c_list[1]),
        ) 
        self.encoder3 = nn.Sequential(
            PVMLayer(c_list[1], c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge: 
            self.scab = DA_Att_Bridge1(c_list, split_att)

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        ) 
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        ) 
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )  
        self.decoder4 = nn.Sequential(
            PVMLayer(c_list[2], c_list[1]),
        )  
        self.decoder5 = nn.Sequential(
            PVMLayer(c_list[1], c_list[0]),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)
        self.is_corr = True
        if self.is_corr:
            self.corr = Corr(nclass=2)
            self.proj = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )
        low_channels = 8
        high_channels = 32
        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))
        self.fuse = nn.Sequential(nn.Conv2d(high_channels + 48, 64, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True))
        

        self.classifier = nn.Conv2d(64, 2, 1, bias=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups # 72
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, need_fp=False, use_corr=False):
        dict_return = {}
        h, w = x.shape[-2:]
        # c_list=[8,16,24,32,48,64]
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2)) # 8
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2)) # 16
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2)) # 24
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2)) # 32
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2)) # 48
        t5 = out # b, c4, H/32, W/32
        
        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5) 
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32        # 64
        t6 = out
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        # print(out2.shape)
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        # print(out1.shape)
        # out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        c1, c4 = out1, out4
        # c1 b 8  192 192
        # c4 b 32 24 24
        if need_fp:
            feats_decode = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = self.classifier(feats_decode)
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            if use_corr:
                proj_feats = self.proj(c4)
                corr_out = self.corr(proj_feats, out)
                corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp
            return dict_return

        feats_decode = self._decode(c1, c4)
        out = self.classifier(feats_decode)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if use_corr:
            proj_feats = self.proj(c4)
            corr_out = self.corr(proj_feats, out)
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            return dict_return
        dict_return = out
        return dict_return
        # return torch.sigmoid(out0)
    def _decode(self, c1, c4):
        # c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        return feature
    def _decode1(self, c1, c4):
        # c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce1(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse1(feature)

        return feature
    
class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(64, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(64, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)
        return out
if __name__ == '__main__':
    model = UltraLight_VM_UNet(num_classes=1,input_channels=4).cuda()
    
    x = torch.randn(2, 4, 384, 384).cuda()
    predict = model(x, need_fp=True, use_corr=True)
    print(predict['out'].shape)  
