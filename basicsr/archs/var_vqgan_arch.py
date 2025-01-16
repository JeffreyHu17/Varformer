
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import get_root_logger
from utils.registry import ARCH_REGISTRY
from .quant import VectorQuantizer2
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout): # conv_shortcut=False,  # conv_shortcut: always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h

class ResnetBlock2(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels_in, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels_out, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.act = nn.SiLU(inplace=True)
        if channels_in != channels_out:
            self.residual_func = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.residual_func = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + self.residual_func(residual)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        
        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels, 3*in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)
        
        # compute attention
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()     # B,HW,C
        k = k.view(B, C, H * W).contiguous()    # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)  
        w = F.softmax(w, dim=2)
        
        
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  
        h = torch.bmm(v, w) 
        h = h.view(B, C, H, W).contiguous()
        
        return x + self.proj_out(h)


class AnchorCrossAttnBlock(nn.Module):
    def __init__(self, in_channels1,in_channels2):
        super().__init__()
        self.C = in_channels1

        self.w_ratio = int(in_channels1) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels1, in_channels1, kernel_size=1, stride=1, padding=0)
    
        self.norm_q = Normalize(in_channels1)
        self.q = torch.nn.Conv2d(in_channels1, in_channels1, kernel_size=1, stride=1, padding=0)
    
        self.norm_kv = Normalize(in_channels2)
        self.kv = torch.nn.Conv2d(in_channels2, 2*in_channels1, kernel_size=1, stride=1, padding=0)
    
        self.s = 4
        self.norm_a = Normalize(in_channels1+in_channels2)
        self.a = torch.nn.Conv2d(in_channels1+in_channels2, in_channels1, kernel_size=self.s, stride=self.s, padding=0)
    
    def forward(self, x,p):
        q = self.q(self.norm_q(x))
        kv = self.kv(self.norm_kv(p))

        t = torch.cat([x,p],1)
        a = self.a(self.norm_a(t))
        B, _, H, W = kv.shape  # should be B,3C,H,W
        C = self.C
        q = q.reshape(B,C, H, W)
        k, v = kv.reshape(B, 2, C, H, W).unbind(1)
        a = a.reshape(B, C, H//self.s, W//self.s)

        
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()     # B,HW,C
        a = a.view(B, C, H * W//(self.s*self.s)).contiguous() # B,C,HW/16
             
        k = k.view(B, C, H * W).contiguous()    # B,C,HW
        w1 = torch.bmm(q, a).mul_(self.w_ratio)  
        w1 = F.softmax(w1, dim=2)

        
        a = a.permute(0, 2, 1).contiguous()     # B,HW/16,C   
        w2 = torch.bmm(a, k).mul_(self.w_ratio)  # B,HW/16,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
        w2 = F.softmax(w2, dim=2)        

        
        v = v.view(B, C, H * W).contiguous()
        w1 = w1.permute(0, 2, 1).contiguous()  
        w2 = w2.permute(0, 2, 1).contiguous()  
        h = torch.bmm(v, w2)  #  B,HW,HW/16         
        h = torch.bmm(h, w1)
        h = h.view(B, C, H, W).contiguous()
        
        return x + self.proj_out(h)

class AnchorCrossAttnBlock2(nn.Module):
    def __init__(self, in_channels1,in_channels2,s=8,k_s=3):
        super().__init__()
        self.C = in_channels1

        self.w_ratio = int(in_channels1) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels1, in_channels1, kernel_size=1, stride=1, padding=0)
    
        self.norm_q = Normalize(in_channels1)
        self.q = torch.nn.Conv2d(in_channels1, in_channels1, kernel_size=1, stride=1, padding=0)
    
        self.norm_kv = Normalize(in_channels2)
        self.kv = torch.nn.Conv2d(in_channels2, 2*in_channels1, kernel_size=1, stride=1, padding=0)
    
        self.s = s
        self.norm_a = Normalize(in_channels1+in_channels2)
        self.a = torch.nn.Conv2d(in_channels1+in_channels2, in_channels1, kernel_size=k_s, stride=self.s, padding=0)
    
    def forward(self, x,p):
        q = self.q(self.norm_q(x))
        kv = self.kv(self.norm_kv(p))

        t = torch.cat([x,p],1)
        a = self.a(self.norm_a(t))
        B, _, H, W = kv.shape  # should be B,3C,H,W
        C = self.C
        q = q.reshape(B,C, H, W)
        k, v = kv.reshape(B, 2, C, H, W).unbind(1)
        
        a = a.reshape(B, C, H//self.s, W//self.s)

        
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()     # B,HW,C
        a = a.view(B, C, H * W//(self.s*self.s)).contiguous() # B,C,HW/16
             
        k = k.view(B, C, H * W).contiguous()    # B,C,HW
        w1 = torch.bmm(q, a).mul_(self.w_ratio)  
        w1 = F.softmax(w1, dim=2)

        
        a = a.permute(0, 2, 1).contiguous()     # B,HW/16,C   
        w2 = torch.bmm(a, k).mul_(self.w_ratio)  # B,HW/16,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
        w2 = F.softmax(w2, dim=2)        

        
        v = v.view(B, C, H * W).contiguous()
        w1 = w1.permute(0, 2, 1).contiguous()  
        w2 = w2.permute(0, 2, 1).contiguous()  
        h = torch.bmm(v, w2)  #  B,HW,HW/16         
        h = torch.bmm(h, w1)
        h = h.view(B, C, H, W).contiguous()
        
        return x + self.proj_out(h)

def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class VarDecoder2(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,  # in_channels: raw img channels
        z_channels, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z):

        dec_res_feats = {}
        
        
        # z to block_in
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        dec_res_feats['z_quant'] = h
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

            if i_level > 0:
                dec_res_feats['Level_%d' % i_level]=h
            
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h, dec_res_feats
    


class VarEncoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, en_f = False, out_feature_list = None):
        enc_feat_dict = {}
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

            if en_f == True and i_level in out_feature_list:
                enc_feat_dict[str(h.shape[-1])] = h.clone()
                

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        if en_f == True:
            enc_feat_dict['middle'] = h.clone()

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        if en_f:
            return h, enc_feat_dict
        else:
            return h


@ARCH_REGISTRY.register()
class VarVQAutoEncoder2(nn.Module):
    def __init__(
            self, vocab_size=4096, z_channels=32, ch=160, dropout=0.0,
            beta=0.25,              
            using_znorm=False,      
            quant_conv_ks=3,        
            quant_resi=0.5,         
            share_quant_resi=4,     
            default_qresi_counts=0, 
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 
            test_mode=True, model_path=None,):
        
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = VarEncoder(double_z=False, **ddconfig)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.decoder = VarDecoder2(**ddconfig)
        
        if model_path is not None:
            key = self.load_state_dict(torch.load(model_path, map_location='cpu'))
            

            


    def forward(self, x, ret_usages=False):
        

        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(x)), ret_usages=ret_usages)
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))
        return h.add_(1).mul_(0.5), dec_res_feats, vq_loss, usages
    

    def fhat_to_img(self, f_hat: torch.Tensor):
        h, _ = self.decoder(self.post_quant_conv(f_hat))

        return h.clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def img_to_encoder_out(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        f = self.quant_conv(self.encoder(x))
        return f

    def img_to_encoder_out_get_f(self, x, out_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.encoder(x, en_f = True, out_feature_list = out_list)
        f = self.quant_conv(x)
        return f, out_feature_list
            
    def encoder_out_to_idxBl(self, f, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)



    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]



    def idxBl_to_img2(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img2(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one, layer_ls=layer_ls)
    
    def embed_to_img2(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True,layer_ls=layer_ls)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False,layer_ls=layer_ls)]

    
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    
    

from archs.RAC import WPMoudle4_norm, RACMoudle2
class VarMainEncoder__varformer2_norm(nn.Module):  
    def __init__(
        self, *, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()


       
        self.DAE = nn.ModuleDict()
        self.AFT = nn.ModuleDict()
        
        self.cat_rec_syn =  nn.ModuleDict()
        channels_prev = 0
        self.cat_rec_syn_dims = [160,160,320,320,640]
        
        for i_level in range(self.num_resolutions):
            
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            
            if i_level != self.num_resolutions - 1:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 2**(self.num_resolutions - 1-i_level), depth=2, num_heads=8, window_size=4)
            else:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 1, depth=2, num_heads=8, window_size=4)

            previous_offset_channel = channels_prev
            self.DAE['Level_%d' % i_level] = RACMoudle2(dim=block_in, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), depth=2, num_heads=8, window_size=4)
            self.AFT['Level_%d' % i_level] = AnchorCrossAttnBlock(block_in, block_in)
            channels_prev = block_in


            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        


        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
    
    def forward(self,  x, prompt_hat, enc_feat_dict_var, en_f = False, out_feature_list = None):
        enc_feat_dict = {}
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            
            if i_level != 0:
                var_prior = self.cat_rec_syn['Level_%d' % i_level](enc_feat_dict_var['Level_%d' % i_level],prompt_hat)
                
                h = self.DAE['Level_%d' % i_level](h, var_prior)
                h  = self.AFT['Level_%d' % i_level](h, var_prior)

            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            enc_feat_dict['Level_%d' % i_level] = h.clone()
            if i_level != self.num_resolutions - 1:
                
                h = self.down[i_level].downsample(h)
            

                                     
            
                

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        if en_f == True:
            enc_feat_dict['middle'] = h.clone()

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        if en_f:
            return h, enc_feat_dict
        else:
            return h





from archs.RAC import WPMoudle4_norm, RACMoudle2
class VarMainEncoder__varformer2_norm_kzj(nn.Module):  
    def __init__(
        self, *, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()


       
        self.DAE = nn.ModuleDict()
        self.AFT = nn.ModuleDict()
        
        self.cat_rec_syn2 =  nn.ModuleDict()
        channels_prev = 0
        
        self.cat_rec_syn_dims = [160,160,160,320,320,640] # encoder main
        
        for i_level in range(self.num_resolutions):
            
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            
            if i_level != self.num_resolutions - 1:
                self.cat_rec_syn2['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 2**(self.num_resolutions - 1-i_level), depth=2, num_heads=8, window_size=4)
            else:
                self.cat_rec_syn2['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 1, depth=2, num_heads=8, window_size=4)

            previous_offset_channel = channels_prev
            self.DAE['Level_%d' % i_level] = RACMoudle2(dim=block_in, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), depth=2, num_heads=8, window_size=4)
            self.AFT['Level_%d' % i_level] = AnchorCrossAttnBlock(block_in, block_in)
            channels_prev = block_in


            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        


        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
    
    def forward(self,  x, prompt_hat, en_f = False, out_feature_list = None):
        enc_feat_dict = {}
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            
            if i_level != 0:
                
                var_prior = self.cat_rec_syn2['Level_%d' % i_level](h,prompt_hat)
                
                h = self.DAE['Level_%d' % i_level](h, var_prior)
                h  = self.AFT['Level_%d' % i_level](h, var_prior)

            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            enc_feat_dict['Level_%d' % i_level] = h.clone()
            if i_level != self.num_resolutions - 1:
                
                h = self.down[i_level].downsample(h)
            

                                     
            
                

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        if en_f == True:
            enc_feat_dict['middle'] = h.clone()

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        if en_f:
            return h, enc_feat_dict
        else:
            return h


@ARCH_REGISTRY.register()
class VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2_kzj(nn.Module): 
    def __init__(
            self, vocab_size=4096, z_channels=32, ch=160, dropout=0.0,
            beta=0.25,              
            using_znorm=False,      
            quant_conv_ks=3,        
            quant_resi=0.5,         
            share_quant_resi=4,     
            default_qresi_counts=0, 
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 
            test_mode=True, model_path=None,):
        
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = VarEncoder_2(double_z=False, **ddconfig)

        self.Mainencoder = VarMainEncoder__varformer2_norm_kzj(double_z=False, **ddconfig)




        



        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        

        



        

        

        
        if model_path is not None:
            
            key = self.load_state_dict(torch.load(model_path, map_location='cpu'))
            

            


    def forward(self, x, ret_usages=False):
        

        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(x)), ret_usages=ret_usages)
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))
        return h.add_(1).mul_(0.5), dec_res_feats, vq_loss, usages
    

    def fhat_to_img(self, f_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))

        return h.clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def img_to_encoder_out(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        f = self.quant_conv(self.encoder(x))
        return f

    def img_to_encoder_out_get_f(self, x, out_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.encoder(x, en_f = True, out_feature_list = out_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def img_to_mainencoder_out_get_f(self, x, prompt_hat, out_feature_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.Mainencoder(x=x, prompt_hat=prompt_hat, en_f = True, out_feature_list = out_feature_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def encoder_out_to_idxBl(self, f, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)



    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]



    def idxBl_to_img2(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img2(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one, layer_ls=layer_ls)
    
    def embed_to_img2(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True,layer_ls=layer_ls)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False,layer_ls=layer_ls)]

    
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    

@ARCH_REGISTRY.register()
class VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2(nn.Module): 
    def __init__(
            self, vocab_size=4096, z_channels=32, ch=160, dropout=0.0,
            beta=0.25,              
            using_znorm=False,      
            quant_conv_ks=3,        
            quant_resi=0.5,         
            share_quant_resi=4,     
            default_qresi_counts=0, 
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 
            test_mode=True, model_path=None,):
        
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = VarEncoder_2(double_z=False, **ddconfig)

        self.Mainencoder = VarMainEncoder__varformer2_norm(double_z=False, **ddconfig)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if model_path is not None:
            
            key = self.load_state_dict(torch.load(model_path, map_location='cpu'))
 
    def forward(self, x, ret_usages=False):
        

        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(x)), ret_usages=ret_usages)
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))
        return h.add_(1).mul_(0.5), dec_res_feats, vq_loss, usages
    

    def fhat_to_img(self, f_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))

        return h.clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def img_to_encoder_out(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        f = self.quant_conv(self.encoder(x))
        return f

    def img_to_encoder_out_get_f(self, x, out_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.encoder(x, en_f = True, out_feature_list = out_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def img_to_mainencoder_out_get_f(self, x, prompt_hat, enc_feat_dict_var, out_feature_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.Mainencoder(x=x, prompt_hat=prompt_hat, enc_feat_dict_var=enc_feat_dict_var, en_f = True, out_feature_list = out_feature_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def encoder_out_to_idxBl(self, f, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)



    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]



    def idxBl_to_img2(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img2(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one, layer_ls=layer_ls)
    
    def embed_to_img2(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True,layer_ls=layer_ls)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False,layer_ls=layer_ls)]

    
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    

class VarEncoder_2(nn.Module):  
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, en_f = False, out_feature_list = None):
        enc_feat_dict = {}
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            
            enc_feat_dict['Level_%d' % i_level] = h.clone()
            
            if i_level != self.num_resolutions - 1:
                
                h = self.down[i_level].downsample(h)
            
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        if en_f == True:
            enc_feat_dict['middle'] = h.clone()

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        if en_f:
            return h, enc_feat_dict
        else:
            return h


class VarDecoder2_2(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,  # in_channels: raw img channels
        z_channels, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z):

        dec_res_feats = {}
        
        
        # z to block_in
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        dec_res_feats['z_quant'] = h
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            dec_res_feats['Level_%d' % i_level]=h
            
            if i_level != 0:
                h = self.up[i_level].upsample(h)

            
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h, dec_res_feats
   


class MainDecoder_varformer2(nn.Module):
    def __init__(
        self, *, ch=160, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=1,
        dropout=0.0, in_channels=3,  # in_channels: raw img channels
        z_channels, using_sa=True, using_mid_sa=True,
        connect_list=['32', '64', '128'],
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        self.beta = torch.tensor(1.0) 
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        

       
        self.DAE = nn.ModuleDict()
        self.AFT = nn.ModuleDict()
        
        self.theta = nn.Parameter(torch.ones(self.num_resolutions))
        
        self.cat_rec_syn =  nn.ModuleDict()
        self.cat_rec_syn_dims = [160,160,320,320,640]
        channels_prev = 0
        for i_level in reversed(range(self.num_resolutions)):
            channels_prev = block_in 

            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * in_ch_mult[i_level]
            block_in = ch * ch_mult[i_level]

            if i_level != self.num_resolutions - 1:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 2**(self.num_resolutions - 1-i_level), depth=2, num_heads=4, window_size=4)
            else:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 1, depth=2, num_heads=4, window_size=4)

            previous_offset_channel = channels_prev
            self.DAE['Level_%d' % i_level] = RACMoudle2(dim=block_in, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), depth=2, num_heads=4, window_size=4)
            self.AFT['Level_%d' % i_level] = AnchorCrossAttnBlock(block_in, block_in)
            channels_prev = block_in


            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)


        # encoder res --> decoder

        self.connect_list = connect_list
        self.channels = {

            '32': 320,
            '64': 160,
            '128': 160,
            'middle': 640,
        }

    
    def forward(self, prompt_hat, x_encoder_out, enc_feat_dict_vf, enc_feat_dict_var, fuse_list = None):
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(x_encoder_out))))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            

            w = self.beta * torch.sigmoid(self.theta[i_level])
            
            h = w * h+ (1-w)*enc_feat_dict_vf['Level_%d' % i_level]

            # h = self.norm_0(h)
            if i_level != 0:
                var_prior = self.cat_rec_syn['Level_%d' % i_level](enc_feat_dict_var['Level_%d' % i_level],prompt_hat)
                h = self.DAE['Level_%d' % i_level](h, var_prior)
                h = self.AFT['Level_%d' % i_level](h, var_prior)

            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)  
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h




from archs.mainDecoder import TWAM, TWAM2, TextureWarpingModule

class MainDecoder_varformer2_patial_gres_2_opt(nn.Module):
    def __init__(
        self, *, ch=160, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=1,
        dropout=0.0, in_channels=3,  # in_channels: raw img channels
        z_channels, using_sa=True, using_mid_sa=True,
        connect_list=['32', '64', '128'],dec_adjust=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.dec_adjust = dec_adjust

        self.beta = torch.tensor(1.0) 
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        

       
        self.DAE = nn.ModuleDict()
        self.AFT = nn.ModuleDict()
        
        self.theta = nn.Parameter(torch.ones(self.num_resolutions))
        
        self.cat_rec_syn =  nn.ModuleDict()
        self.cat_rec_syn_bf  = nn.ModuleDict()
        self.cat_rec_syn_bf_nm  = nn.ModuleDict()

        self.cat_rec_syn_dims = [160,160,320,320,640]
        self.cat_rec_syn_dims_h = [160,160,320,320,640]
        channels_prev = 0


       
        if self.dec_adjust:
            self.off_attn = nn.ModuleDict()
            self.align_func_dict = nn.ModuleDict()
            self.cat_convs = nn.ModuleDict()

        for i_level in reversed(range(self.num_resolutions)):
            channels_prev = block_in 
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * in_ch_mult[i_level]
            block_in = ch * ch_mult[i_level]

            
            self.cat_rec_syn_bf_nm['Level_%d' % i_level] = Normalize(self.cat_rec_syn_dims[i_level]+self.cat_rec_syn_dims_h[i_level])
            self.cat_rec_syn_bf['Level_%d' % i_level] = nn.Conv2d(in_channels=self.cat_rec_syn_dims[i_level]+self.cat_rec_syn_dims_h[i_level], out_channels=self.cat_rec_syn_dims[i_level], kernel_size=1, stride=1, padding=0)             
            
            
            if i_level != self.num_resolutions - 1:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 2**(self.num_resolutions - 1-i_level), depth=2, num_heads=4, window_size=4)
            else:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 1, depth=2, num_heads=4, window_size=4)

            previous_offset_channel = channels_prev
            self.DAE['Level_%d' % i_level] = RACMoudle2(dim=block_in, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), depth=2, num_heads=4, window_size=4)
            self.AFT['Level_%d' % i_level] = AnchorCrossAttnBlock(block_in, block_in)
            channels_prev = block_in


            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order
        


            if self.dec_adjust:
                if i_level == self.num_resolutions - 1:
                    self.off_attn['af_quant'] = TWAM(channels_prev, num_heads=ch_mult[i_level])

                    self.align_func_dict['af_quant'] = \
                        TextureWarpingModule(
                            channel=channels_prev,
                            cond_channels=channels_prev,
                            deformable_groups=4,
                            previous_offset_channel=0)

                if i_level > 0:
                    previous_offset_channel = channels_prev
                    self.off_attn['Level_%d' % i_level] = TWAM(block_out, num_heads=ch_mult[i_level-1])
                    
                    self.align_func_dict['Level_%d' % i_level] = \
                        TextureWarpingModule(
                            channel=block_out,
                            cond_channels=block_out,
                            deformable_groups=4,
                            previous_offset_channel=previous_offset_channel)
                    self.cat_convs['Level_%d' % i_level] = nn.Conv2d(in_channels=block_out*2, out_channels=block_in, kernel_size=1, stride=1, padding=0)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)


        # encoder res --> decoder

        self.connect_list = connect_list
        self.channels = {

            '32': 320,
            '64': 160,
            '128': 160,
            'middle': 640,
        }

        self.input_w = nn.Parameter(torch.tensor(0.5))
        self.conv1__w = nn.Conv2d(1, 1, kernel_size=1)
        self.softmax__w = nn.Softmax(dim=1)

    
    def forward(self, inp_B3HW, prompt_hat, x_encoder_out, enc_feat_dict_vf, enc_feat_dict_var, fuse_list = None):

        # z to block_in
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(x_encoder_out))))
        if self.dec_adjust:
            x_d, x_g = self.off_attn['af_quant'](h, enc_feat_dict_var['z_quant'])
            h, offset = self.align_func_dict['af_quant'](x_d, x_g)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            

            w = self.beta * torch.sigmoid(self.theta[i_level])
            
            h = w * h+ (1-w)*enc_feat_dict_vf['Level_%d' % i_level]

            # h = self.norm_0(h)
            if i_level != 0:
                vg = self.cat_rec_syn_bf_nm['Level_%d' % i_level](torch.cat((enc_feat_dict_var['Level_%d' % i_level],h),1))
                vg = self.cat_rec_syn_bf['Level_%d' % i_level](vg)
                
                var_prior = self.cat_rec_syn['Level_%d' % i_level](vg,prompt_hat)

                h = self.DAE['Level_%d' % i_level](h, var_prior)
                h = self.AFT['Level_%d' % i_level](h, var_prior)

            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)  
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if self.dec_adjust:
                    upsample_offset = F.interpolate(offset, scale_factor=2, align_corners=False, mode='bilinear') * 2
                    x_d, x_g = self.off_attn['Level_%d' % i_level](h, enc_feat_dict_var['Level_%d' % (i_level-1)]) # F_o_d  F_o_g
                    warp_feat, offset = self.align_func_dict['Level_%d' % i_level](x_d, x_g, previous_offset=upsample_offset)  # F_out Offset
                    h = torch.cat([x_d, warp_feat], dim=1)
                    h = self.cat_convs['Level_%d' % i_level](h)

        w_z = self.softmax__w(self.conv1__w(enc_feat_dict_vf['weighted_z'])).squeeze(1)
        res = torch.einsum('nhw,nchw->nchw', w_z, inp_B3HW)
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        h = h*(1-self.input_w) + res *self.input_w 
        return h





@ARCH_REGISTRY.register()
class VarVQAutoEncoder(nn.Module): 
    def __init__(
            self, vocab_size=4096, z_channels=32, ch=160, dropout=0.0,
            beta=0.25,              
            using_znorm=False,      
            quant_conv_ks=3,        
            quant_resi=0.5,         
            share_quant_resi=4,     
            default_qresi_counts=0, 
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 
            test_mode=True, model_path=None,):
        
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = VarEncoder_2(double_z=False, **ddconfig)

        self.Mainencoder = VarMainEncoder__varformer2_norm_patial_gres(double_z=False, **ddconfig)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)



        self.decoder = VarDecoder2_2(**ddconfig)

        

        
        if model_path is not None:
            
            key = self.load_state_dict(torch.load(model_path, map_location='cpu'))
            

            


    def forward(self, x, ret_usages=False):
        

        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(x)), ret_usages=ret_usages)
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))
        return h.add_(1).mul_(0.5), dec_res_feats, vq_loss, usages
    

    def fhat_to_img(self, f_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))

        return h.clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def img_to_encoder_out(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        f = self.quant_conv(self.encoder(x))
        return f

    def img_to_encoder_out_get_f(self, x, out_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.encoder(x, en_f = True, out_feature_list = out_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def img_to_mainencoder_out_get_f(self, x, prompt_hat, enc_feat_dict_var, out_feature_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.Mainencoder(x=x, prompt_hat=prompt_hat, enc_feat_dict_var=enc_feat_dict_var, en_f = True, out_feature_list = out_feature_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def encoder_out_to_idxBl(self, f, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)



    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]



    def idxBl_to_img2(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img2(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one, layer_ls=layer_ls)
    
    def embed_to_img2(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True,layer_ls=layer_ls)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False,layer_ls=layer_ls)]

    
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    


@ARCH_REGISTRY.register()
class VarVQAutoEncoder_varRec(nn.Module): 
    def __init__(
            self, vocab_size=4096, z_channels=32, ch=160, dropout=0.0,
            beta=0.25,              
            using_znorm=False,      
            quant_conv_ks=3,        
            quant_resi=0.5,         
            share_quant_resi=4,     
            default_qresi_counts=0, 
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 
            test_mode=True, model_path=None,):
        
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = VarEncoder_2(double_z=False, **ddconfig)

        



        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)

        self.decoder = VarDecoder2_2(**ddconfig)

        
        if model_path is not None:
            
            key = self.load_state_dict(torch.load(model_path, map_location='cpu'))
            

            


    def forward(self, x, ret_usages=False):
        

        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(x)), ret_usages=ret_usages)
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))
        return h.add_(1).mul_(0.5), dec_res_feats, vq_loss, usages
    

    def fhat_to_img(self, f_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))

        return h.clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def img_to_encoder_out(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        f = self.quant_conv(self.encoder(x))
        return f

    def img_to_encoder_out_get_f(self, x, out_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.encoder(x, en_f = True, out_feature_list = out_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def img_to_mainencoder_out_get_f(self, x, prompt_hat, enc_feat_dict_var, out_feature_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.Mainencoder(x=x, prompt_hat=prompt_hat, enc_feat_dict_var=enc_feat_dict_var, en_f = True, out_feature_list = out_feature_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def encoder_out_to_idxBl(self, f, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)



    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]



    def idxBl_to_img2(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img2(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one, layer_ls=layer_ls)
    
    def embed_to_img2(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True,layer_ls=layer_ls)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False,layer_ls=layer_ls)]

    
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    

from archs.RAC import WPMoudle4_norm, RACMoudle2
class VarMainEncoder__varformer2_norm_patial_gres(nn.Module):  
    def __init__(
        self, *, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()


       
        self.DAE = nn.ModuleDict()
        self.AFT = nn.ModuleDict()
        
        self.cat_rec_syn =  nn.ModuleDict()
        self.cat_rec_syn_bf  = nn.ModuleDict()
        self.cat_rec_syn_bf_nm  = nn.ModuleDict()
        channels_prev = 0
        self.cat_rec_syn_dims = [160,160,320,320,640]
        self.cat_rec_syn_dims_h = [160,160,160,320,320,640]
        for i_level in range(self.num_resolutions):
            
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            
            self.cat_rec_syn_bf_nm['Level_%d' % i_level] = Normalize(self.cat_rec_syn_dims[i_level]+self.cat_rec_syn_dims_h[i_level])
            self.cat_rec_syn_bf['Level_%d' % i_level] = nn.Conv2d(in_channels=self.cat_rec_syn_dims[i_level]+self.cat_rec_syn_dims_h[i_level], out_channels=self.cat_rec_syn_dims[i_level], kernel_size=1, stride=1, padding=0) 
            
            if i_level != self.num_resolutions - 1:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 2**(self.num_resolutions - 1-i_level), depth=2, num_heads=8, window_size=4)
            else:
                self.cat_rec_syn['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 1, depth=2, num_heads=8, window_size=4)

            previous_offset_channel = channels_prev
            self.DAE['Level_%d' % i_level] = RACMoudle4(dim=block_in, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), depth=2, num_heads=8, window_size=4)
            self.AFT['Level_%d' % i_level] = AnchorCrossAttnBlock(block_in, block_in)
            channels_prev = block_in


            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        


        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
    
    def forward(self,  x, prompt_hat, enc_feat_dict_var, en_f = False, out_feature_list = None):
        enc_feat_dict = {}
        # downsampling
        h = self.conv_in(x)
        h1 = None
        for i_level in range(self.num_resolutions):            
        
            if i_level == 0:
                h1 = h           
            else:
                
                vg = self.cat_rec_syn_bf_nm['Level_%d' % i_level](torch.cat((enc_feat_dict_var['Level_%d' % i_level],h),1))
                vg = self.cat_rec_syn_bf['Level_%d' % i_level](vg)
                
                var_prior = self.cat_rec_syn['Level_%d' % i_level](vg,prompt_hat)
                

                if i_level == 1:
                    
                    h, w_z = self.DAE['Level_%d' % i_level](h, var_prior)
                    
                    w_z = F.interpolate(w_z, scale_factor=2, mode='nearest')
                    
                    
                    enc_feat_dict['weighted_z'] = w_z #.squeeze(1) #torch.einsum('nhw,nchw->nchw', w_z.squeeze(1), h1)
                else:
                    h,_ = self.DAE['Level_%d' % i_level](h, var_prior)

                h  = self.AFT['Level_%d' % i_level](h, var_prior)


            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            enc_feat_dict['Level_%d' % i_level] = h.clone()
            if i_level != self.num_resolutions - 1:
                
                h = self.down[i_level].downsample(h)
            
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        if en_f == True:
            enc_feat_dict['middle'] = h.clone()

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        if en_f:
            return h, enc_feat_dict
        else:
            return h



from archs.RAC import WPMoudle4_norm, RACMoudle2, RACMoudle3
class VarMainEncoder__varformer2_norm_kzj_patial_gres(nn.Module):  
    def __init__(
        self, *, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()


       
        self.DAE = nn.ModuleDict()
        self.AFT = nn.ModuleDict()
        
        self.cat_rec_syn2 =  nn.ModuleDict()
        channels_prev = 0
        
        self.cat_rec_syn_dims = [160,160,160,320,320,640] # encoder main
        
        for i_level in range(self.num_resolutions):
            
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            
            if i_level != self.num_resolutions - 1:
                self.cat_rec_syn2['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 2**(self.num_resolutions - 1-i_level), depth=2, num_heads=8, window_size=4)
            else:
                self.cat_rec_syn2['Level_%d' % i_level] = WPMoudle4_norm(dim_in=self.cat_rec_syn_dims[i_level], dim_out=10, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), out_dim = block_in, up_r= 1, depth=2, num_heads=8, window_size=4)

            previous_offset_channel = channels_prev
            self.DAE['Level_%d' % i_level] = RACMoudle3(dim=block_in, input_resolution=(16*2**(self.num_resolutions - 1-i_level),16*2**(self.num_resolutions - 1-i_level)), depth=2, num_heads=8, window_size=4)
            self.AFT['Level_%d' % i_level] = AnchorCrossAttnBlock(block_in, block_in)
            channels_prev = block_in


            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        


        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
    
    def forward(self,  x, prompt_hat, en_f = False, out_feature_list = None):
        enc_feat_dict = {}
        # downsampling
        h = self.conv_in(x)
        h1 = None
        for i_level in range(self.num_resolutions):

            if i_level == 0:
                h1 = h           
            else:                
                var_prior = self.cat_rec_syn2['Level_%d' % i_level](h,prompt_hat)
                if i_level == 1:
                    
                    h, w_z = self.DAE['Level_%d' % i_level](h, var_prior)                    
                    w_z = F.interpolate(w_z, scale_factor=2, mode='nearest')                    
                    enc_feat_dict['weighted_z_layer1'] = torch.einsum('nhw,nchw->nchw', w_z.squeeze(1), h1)
                else:
                    h,_ = self.DAE['Level_%d' % i_level](h, var_prior)

                h  = self.AFT['Level_%d' % i_level](h, var_prior)

            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            enc_feat_dict['Level_%d' % i_level] = h.clone()
            if i_level != self.num_resolutions - 1:
                
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        if en_f == True:
            enc_feat_dict['middle'] = h.clone()

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        if en_f:
            return h, enc_feat_dict
        else:
            return h


@ARCH_REGISTRY.register()
class VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2_kzj_patial_gres(nn.Module): 
    def __init__(
            self, vocab_size=4096, z_channels=32, ch=160, dropout=0.0,
            beta=0.25,              
            using_znorm=False,      
            quant_conv_ks=3,        
            quant_resi=0.5,         
            share_quant_resi=4,     
            default_qresi_counts=0, 
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), 
            test_mode=True, model_path=None,):
        
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = VarEncoder_2(double_z=False, **ddconfig)

        self.Mainencoder = VarMainEncoder__varformer2_norm_kzj_patial_gres(double_z=False, **ddconfig)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)

        if model_path is not None:
            
            key = self.load_state_dict(torch.load(model_path, map_location='cpu'))


    def forward(self, x, ret_usages=False):
        

        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(x)), ret_usages=ret_usages)
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))
        return h.add_(1).mul_(0.5), dec_res_feats, vq_loss, usages
    

    def fhat_to_img(self, f_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))

        return h.clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def img_to_encoder_out(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        f = self.quant_conv(self.encoder(x))
        return f

    def img_to_encoder_out_get_f(self, x, out_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.encoder(x, en_f = True, out_feature_list = out_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def img_to_mainencoder_out_get_f(self, x, prompt_hat, out_feature_list, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False):
        x, out_feature_list = self.Mainencoder(x=x, prompt_hat=prompt_hat, en_f = True, out_feature_list = out_feature_list)
        f = self.quant_conv(x)
        return f, out_feature_list

    def encoder_out_to_idxBl(self, f, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)



    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]



    def idxBl_to_img2(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            p_si = self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn)
            ms_h_BChw.append(p_si)
            
        return self.embed_to_img2(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one, layer_ls=layer_ls)
    
    def embed_to_img2(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False, layer_ls=[]) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True,layer_ls=layer_ls)))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat2(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False,layer_ls=layer_ls)]

    
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]))[0].clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat))[0].clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    


