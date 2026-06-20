import math
import os
import einops
from peft import LoraConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Vision_encoder.Vit4AD import MAETS_AD, VitTS_AD
# from loss.head import AD_classifier,TS_Reconstruction,AD_Reconstruction_P

import os
# 获取当前文件所在目录的父目录（项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
vision_PATH = os.path.join(BASE_DIR, 'checkpoints', 'weight_v')

class V_model(nn.Module):
    def __init__(self, vision_name=None, unpatch=True, MAX_L=5000, finetune_type='none',
                 use_vectorized_fold=False, **kwargs):
        """
        Vision Encoder based on MAE/ViT.

        Args:
            vision_name: Name of the vision weight file
            unpatch: Whether to use unpatch operation
            MAX_L: Maximum sequence length
            finetune_type: Fine-tuning strategy for ViT
                - 'none': Fully frozen (default, as per paper)
                - 'ln': Only LayerNorm trainable
                - 'bias': Only bias trainable
                - 'full': All parameters trainable
                - 'mlp': Only MLP layers trainable
                - 'attn': Only attention layers trainable
            use_vectorized_fold: If True, use vectorized fold_image (faster, fixed T_sqrt=True).
                                 If False, use original fold_image with period detection.
                                 Default: False for backward compatibility.

        Note: Paper specifies "the encoder of the frozen MAE", so default is 'none'.
        """
        super().__init__()

        self.use_vectorized_fold = use_vectorized_fold

        vision_weight = os.path.join(vision_PATH,vision_name)
        if 'vit' in vision_name:
            self.encode_image= VitTS_AD(vision_weight)
            self.config = self.encode_image.config
            self.patch_size = self.config.patch_size
            self.hidden_size = self.config.hidden_size
        elif 'mae' in vision_name:
            self.encode_image= MAETS_AD(vision_weight)
            self.config = self.encode_image.config
            self.patch_size = self.config['patch_size']
            self.hidden_size = self.config['embed_dim']

        self.MAX_L=MAX_L
        self._freeze_layers()

        # Apply fine-tuning strategy (default: 'none' = fully frozen, as per paper)
        if finetune_type != 'full':
            for n, param in self.encode_image.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    def _freeze_layers(self):
        # freeze all the parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def fold_image(self, images, P_L, p_values, img_size=224, T_sqrt=False):
        """
        Fold time series images into 2D representation.

        When use_vectorized_fold=True (set in __init__), this method automatically
        uses the vectorized implementation which is ~150x faster but requires T_sqrt=True.
        When use_vectorized_fold=False, uses the original implementation with period detection.

        Args:
            images: (B, 3, Num, W) input images
            P_L: Period lengths for each sample (only used when T_sqrt=False)
            p_values: Padding values, tuple of (C_ts, 3) tensors
            img_size: Output image size, default 224
            T_sqrt: If True, use sqrt(T) for period. If False, use detected period P_L.
                   Note: Vectorized version always uses T_sqrt=True internally.

        Returns:
            img_2d_batch: (B, 3, img_size, img_size) folded 2D images
            results_size_out: Size parameters for unfold_image
        """
        # 如果启用向量化版本，直接调用（向量化版本固定 T_sqrt=True）
        if self.use_vectorized_fold:
            return self.fold_image_vectorized(images, p_values, img_size=img_size)

        # 原始实现
        B, C, Num, W = images.shape
        results_2d = []
        results_size_out = []
        step_size = self.patch_size
        for b in range(B):
            P = int(P_L[b])
            image=images[b]
            p_value=p_values[b]
            T = W // step_size

            if T_sqrt:
                T_p = max(int(round(math.sqrt(T))), 1)
            else:
                T_p = P // step_size if P// step_size  > 1 else max(int(round(math.sqrt(T))), 1)

            init_h,init_w = T//T_p, T_p
            pad_patch = 0
            if T % T_p != 0:
                pad_patch = (T_p - T % T_p)
                pad_pixels = pad_patch * step_size #2*patchsize
                img_pad = F.pad(image.unsqueeze(0), (0, pad_pixels, 0, 0), "constant", 0).squeeze(0)
                init_h += 1

                img_pad[:, :, W:] = p_value.T[:, :, None]

            else:
                img_pad = image
            img_2d_l=[]
            for j in range(Num):
                img_pad_j=img_pad[:,j,:]
                parts = [img_pad_j[:,i*T_p*step_size:(i+1)*T_p*step_size].unsqueeze(1) for i in range(init_h)]
                img_2d_j = torch.cat(parts,axis=1)
                img_2d_l.append(img_2d_j)
            img_2d = torch.cat(img_2d_l,axis=1)
            img_resized_y = F.interpolate(img_2d.unsqueeze(0), size=(img_size, img_2d.shape[2]), mode='nearest', align_corners=None)
            img_final = F.interpolate(img_resized_y, size=(img_size, img_size), mode='bilinear', align_corners=False)
            results_2d.append(img_final)
            img_size_out = [init_h, init_w, pad_patch, img_size, Num]
            results_size_out.append(img_size_out)

        img_2d_batch = torch.cat(results_2d, dim=0)
        return img_2d_batch,results_size_out

    def fold_image_vectorized(self, images, p_values, img_size=224):
        """
        向量化版本的 fold_image，固定使用 T_sqrt=True。

        同一 batch 内所有样本的 W 相同，T = W//patch_size 相同，
        T_p = sqrt(T) 也相同，因此所有 fold 参数一致，
        可以批量 pad + 重排 + interpolate，无需逐样本循环。

        与 fold_image 的区别：
        - 固定 T_sqrt=True（不使用周期 P）
        - 批量处理，使用 unfold/reshape 代替循环
        - 输出与 fold_image 完全一致，unfold_image 可直接使用

        Args:
            images: (B, 3, Num, W) 输入图像，其中 Num = C_ts * h_size
            p_values: tuple of (C_ts, 3) 或 (B, C_ts, 3) padding 填充值
            img_size: 输出图像尺寸，默认 224

        Returns:
            img_2d_batch: (B, 3, img_size, img_size) fold 后的 2D 图像
            results_size_out: List[List[int]] 每个样本的尺寸参数，供 unfold_image 使用
        """
        B, C_img, Num, W = images.shape
        step_size = self.patch_size
        T = W // step_size
        T_p = max(int(round(math.sqrt(T))), 1)

        init_w = T_p
        init_h_base = T // T_p
        pad_patch = 0

        # 处理 p_values：支持 tuple 或 tensor 输入
        if isinstance(p_values, (tuple, list)):
            # tuple of (C_ts, 3) -> stack to (B, C_ts, 3) -> permute to (B, 3, C_ts, 1)
            p_values_tensor = torch.stack(p_values)  # (B, C_ts, 3)
        else:
            p_values_tensor = p_values  # 假设已经是 (B, C_ts, 3)

        # 转换为 (B, 3, C_ts, 1) 用于广播填充
        # 注意：C_ts 应该等于 Num（当 h_size=1 时）
        p_values_4d = p_values_tensor.permute(0, 2, 1).unsqueeze(-1)  # (B, 3, C_ts, 1)

        if T % T_p != 0:
            pad_patch = T_p - T % T_p
            pad_pixels = pad_patch * step_size
            # 批量 padding: (B, 3, Num, W) -> (B, 3, Num, W+pad_pixels)
            images = F.pad(images, (0, pad_pixels, 0, 0), mode='constant', value=0)
            init_h = init_h_base + 1
            # 填充 padding 区域: (B, 3, C_ts, 1) 广播到 (B, 3, Num, pad_pixels)
            images[:, :, :, W:] = p_values_4d
        else:
            init_h = init_h_base

        # 向量化重排：将每个样本的时间维度折叠成 2D 空间
        # 原始逻辑：对每个 j in Num，切分 W 为 init_h 段，沿高度拼接
        # 使用 unfold 实现无循环版本
        # (B, 3, Num, init_h * T_p * step_size) -> (B, 3, Num, init_h, T_p * step_size)
        img_2d = images.unfold(3, T_p * step_size, T_p * step_size)
        # 调换维度: (B, 3, Num, init_h, T_p*step_size) -> (B, 3, Num*init_h, T_p*step_size)
        img_2d = img_2d.permute(0, 1, 2, 3, 4).reshape(B, C_img, Num * init_h, T_p * step_size)

        # 批量 resize: F.interpolate 需要 (N, C, H, W) 格式
        # img_2d 已经是 (B, 3, Num*init_h, T_p*step_size)，符合要求
        img_resized_y = F.interpolate(img_2d, size=(img_size, img_2d.shape[3]), mode='nearest', align_corners=None)
        img_final = F.interpolate(img_resized_y, size=(img_size, img_size), mode='bilinear', align_corners=False)

        # 构造 size_out（与 fold_image 输出格式一致）
        img_size_out = [init_h, init_w, pad_patch, img_size, Num]
        results_size_out = [img_size_out] * B

        return img_final, results_size_out

    def unfold_image(self, x0,size):
        B, L, D = x0.shape
        
        recovered_list = []
        for i in range(B):
            x = x0[i]
            init_h,init_w,pad,h,Num = map(int, size[i])
            w=h = h//self.patch_size
            
            assert h * w == L
            output = x.transpose(0, 1).view(D, h, w)
            output = F.adaptive_avg_pool2d(output.unsqueeze(0), (init_h*Num, w)).squeeze(0)

            output_up = F.interpolate(
                output.unsqueeze(0), 
                size=(init_h*Num, init_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            patches=[]
            for j in range(Num):
                patch=output_up[:, init_h*j:init_h*(j+1), :].view(D, -1).contiguous()
                if pad>0:
                    patch = patch[:,:-pad]
                patches.append(patch)
            unfold = torch.cat(patches, dim=-1).transpose(0, 1)
            
            recovered_list.append(unfold)
        recovered_list = torch.stack(recovered_list)
        return recovered_list

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.encode_image(hidden_states)
        output_ts_patch =hidden_states[:,1:,:]

        return output_ts_patch,None