import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.common import ResidualGroup
from einops import rearrange
import numbers


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## DRFeed-Forward Network
class FeedForward_DR(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=True):
        super(FeedForward_DR, self).__init__()

        # hidden_features = int(dim * ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        #
        # self.dwconv = nn.Conv2d(
        #     hidden_features * 2,
        #     hidden_features * 2,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     groups=hidden_features * 2,
        #     bias=bias,
        # )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x = self.project_in(x)
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Degradation_Attention(nn.Module):
    def __init__(self, dim, bias=True):
        super(Degradation_Attention, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(dim, 128, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(128, dim, kernel_size=1, bias=bias)

    def forward(self, x, dr):
        b, c, h, w = x.shape

        q = self.qkv(x)
        q = q.view(b, -1, h * w)

        # dr shape: (b, 32, 128)
        # hw, c *** c, 1024
        attn = (q.transpose(-2, -1) @ dr.transpose(-2, -1)) * self.temperature  # b, hw, 1024
        attn = attn.softmax(dim=-1)

        out = attn @ dr  # b, hw, c

        out = out.transpose(-2, -1).view(b, -1, h, w)
        out = self.project_out(out)
        return out

class Degradation_recognition(nn.Module):
    def __init__(self, dim):
        super(Degradation_recognition, self).__init__()
        # LayerNorm_type = "WithBias"
        ffn_expansion_factor = 2.66

        self.attn = Degradation_Attention(dim)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_DR(dim, ffn_expansion_factor)

    def forward(self, x, dr):
        dr = self.attn(x, dr)
        # dr = self.ffn(self.norm2(dr))
        dr = self.ffn(dr)

        return dr 

class Low_rank_transform(nn.Module):
    def __init__(self, inchn=64):
        super(Low_rank_transform, self).__init__()
        c=4

        # self.trans = nn.Conv2d(dim, out_n_feat, kernel_size=1, bias=True)
        # For Channel dimennsion
        self.conv_C = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(inchn, 16 * (inchn), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            # torch.nn.Conv2d(16 * c, 16 * (16 * c), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        # For Height dimennsion
        self.conv_H = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((None, 1)),
            torch.nn.Conv2d(inchn, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            # torch.nn.Conv2d(16 * c, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        # For Width dimennsion
        self.conv_W = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, None)),
            torch.nn.Conv2d(inchn, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            # torch.nn.Conv2d(16 * c, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            torch.nn.Sigmoid()
        )

        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        N_,C_,H_,W_ = x.shape

        res = x
        s0_3_c = self.conv_C(x)

        s0_3_c = s0_3_c.view(N_, 16, -1, 1, 1)

        s0_3_h = self.conv_H(x)
        s0_3_h = s0_3_h.view(N_, 16, 1, -1, 1)

        s0_3_w = self.conv_W(x)
        s0_3_w = s0_3_w.view(N_, 16, 1, 1, -1)

        cube0 = (s0_3_c * s0_3_h * s0_3_w).mean(1)

        x = x * cube0

        return  x

class Degradation_inject(nn.Module):
    def __init__(self, dim, out_n_feat):
        super(Degradation_inject, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.f = nn.Conv2d(dim, out_n_feat, kernel_size=1, bias=True)
        self.g = nn.Conv2d(dim, out_n_feat, kernel_size=1, bias=True)

        self.low_rank_transform1 = Low_rank_transform()
        self.low_rank_transform3 = Low_rank_transform(out_n_feat)

    def forward(self, x, dr):
        res = x
        x = self.low_rank_transform3(x)
        # fg = self.fg_dwconv(self.fg(dr))
        # f, g = fg.chunk(2, dim=1)
        dr = self.conv1(dr)
        f = self.f(self.low_rank_transform1(dr))
        g = self.g(self.low_rank_transform1(dr))

        x = f * x + g

        return x 


class Degradation_representation_learning(nn.Module):
    def __init__(self, in_n_feat, n_feat=64):
        super(Degradation_representation_learning, self).__init__()

        self.mapping = nn.Conv2d(
            in_n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.degradation_recog = Degradation_recognition(n_feat)

        self.inject_degradation = Degradation_inject(n_feat, in_n_feat)

    def forward(self, x, dr):
        feat = x
        x = self.mapping(x)

        degradation_info = self.degradation_recog(x, dr) 

        # inject degradation info
        out = self.inject_degradation(feat, degradation_info) 
        return out, degradation_info 
# 

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x


class DegradationNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            dim=48,
            bias=False,
    ):
        super(DegradationNet, self).__init__()

        self.extractor = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.block1 = DoubleConv(dim, dim)
        self.pool1 = Downsample(dim)

        self.block2 = DoubleConv(dim* 2 ** 1, dim* 2 ** 1)
        self.pool2 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.block3 = DoubleConv(dim* 2 ** 2, dim* 2 ** 2)
        self.pool3 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.bottom = DoubleConv(dim * 2**3,dim * 2**3)

        self.inject_degradation1 = Degradation_inject(64, dim * 1)
        self.inject_degradation2 = Degradation_inject(64, dim * 2)
        self.inject_degradation3 = Degradation_inject(64, dim * 4)

        self.up3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.block_up3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias),
            DoubleConv(dim * 2 ** 2, dim * 2 ** 2),
        )

        self.up2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.block_up2 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias),
            DoubleConv(dim * 2 ** 1, dim * 2 ** 1),
        )

        self.up1 = Upsample(
            int(dim * 2 ** 1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.block_up1 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias),
            DoubleConv(dim * 2 ** 1, dim * 2 ** 1),
        )

        self.refinement = ResidualGroup(n_feat=dim * 2 ** 1, kernel_size=3, reduction=16, act=nn.ReLU(True),
                                        res_scale=1, n_resblocks=1)

        self.reconstruction = nn.Conv2d(
            int(dim * 2 ** 1), out_channels, kernel_size=3, padding=1
        )


    def forward(self, x, x_rep1, x_rep2, x_rep3):
        res = x
        _, _, h, w = x.size()

        x = self.extractor(x)  # b, 64, h, w

        x1_1 = self.block1(x)  # b, 64, h, w
        x1_2 = self.pool1(x1_1)  # b, 64, 128, 128

        x2_1 = self.block2(x1_2)
        x2_2 = self.pool2(x2_1)  # b, 64, 64, 64

        x3_1 = self.block3(x2_2)
        x3_2 = self.pool3(x3_1)  # b, 64, 32, 32

        x_res1 = self.inject_degradation1(x1_1, x_rep1)
        x_res2 = self.inject_degradation2(x2_1, x_rep2)
        x_res3 = self.inject_degradation3(x3_1, x_rep3)

        x_bottom = self.bottom(x3_2)

        x_up3 = self.up3(x_bottom)  # b, 64, 64, 64
        x_up3 = torch.cat([x_res3, x_up3], dim=1)  # b, 128, 32, 32
        x_de3 = self.block_up3(x_up3)

        x_up2 = self.up2(x_de3)  # b, 64, 128, 128
        x_up2 = torch.cat([x_res2, x_up2], dim=1)  # b, 128, 64, 64
        x_de2 = self.block_up2(x_up2)  # b, 64, 128, 128

        x_up1 = self.up1(x_de2)  # b, 64, 128, 128
        x_up1 = torch.cat([x_res1, x_up1], dim=1)  # b, 128, 128, 128
        x_de1 = self.block_up1(x_up1)  # b, 64, 128, 128

        x_out = self.refinement(x_de1)

        x_out = self.reconstruction(x_out) + res

        return x_out


class RestorationNet(nn.Module):
    def __init__(
            self,
            dr,
            channels=48,
            in_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            # num_blocks=[1, 1, 1, 1],
            num_refinement_blocks=3,
            heads=[1, 2, 4, 6],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",  ## Other option 'BiasFree'
            dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(RestorationNet, self).__init__()
        self.dr = dr

        self.extractor = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        self.block1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.pool1 = Downsample(dim)

        self.block2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.pool2 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.block3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.pool3 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4

        self.bottom = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.res_connect1 = Degradation_representation_learning(dim * 1)
        self.res_connect2 = Degradation_representation_learning(dim * 2)
        self.res_connect3 = Degradation_representation_learning(dim * 4)

        self.up3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.block_up3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias),
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.block_up2 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias),
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up1 = Upsample(
            int(dim * 2 ** 1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.block_up1 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias),
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.reconstruction = nn.Conv2d(
            int(dim * 2 ** 1), out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):


        res = x
        _, _, h, w = x.size()

        x = self.extractor(x)

        x1_1 = self.block1(x)  #
        x1_2 = self.pool1(x1_1)

        x2_1 = self.block2(x1_2)
        x2_2 = self.pool2(x2_1)

        x3_1 = self.block3(x2_2)
        x3_2 = self.pool3(x3_1)

        x_res1, x_rep1 = self.res_connect1(x1_1, self.dr)
        x_res2, x_rep2 = self.res_connect2(x2_1, self.dr)
        x_res3, x_rep3 = self.res_connect3(x3_1, self.dr)

        x_bottom = self.bottom(x3_2)

        x_up3 = self.up3(x_bottom)
        x_up3 = torch.cat([x_res3, x_up3], dim=1)
        x_de3 = self.block_up3(x_up3)

        x_up2 = self.up2(x_de3)
        x_up2 = torch.cat([x_res2, x_up2], dim=1)
        x_de2 = self.block_up2(x_up2)

        x_up1 = self.up1(x_de2)
        x_up1 = torch.cat([x_res1, x_up1], dim=1)
        x_de1 = self.block_up1(x_up1)

        x_out = self.refinement(x_de1)

        x_out = self.reconstruction(x_out)

        return x_out, x_rep1, x_rep2, x_rep3


class NDRN(nn.Module):
    def __init__(self):
        super(NDRN, self).__init__()
        self.degradation_representations = nn.Parameter(torch.randn(1, 32, 128), requires_grad=True)

        self.degradation_net = DegradationNet()
        self.restoration_net = RestorationNet(self.degradation_representations)

    def restoration_process(self, LQ_img):
        (
            HQ_img,
            self.x_rep1,
            self.x_rep2,
            self.x_rep3,
        ) = self.restoration_net(LQ_img)
        return HQ_img

    def degradation_process(self, HQ_img):
        LQ_img = self.degradation_net(
            HQ_img, self.x_rep1, self.x_rep2, self.x_rep3
        )
        return LQ_img


if __name__ == "__main__":
    HF = torch.randn(1, 3, 64, 64)
    model = NDRN()
    out = model.restoration_process(HF)
    out = model.degradation_process(HF)
    print(out.shape)
    # for k, v in model.named_parameters():
    #     # if v.requires_grad:
    #     # if k in ['gate.proj.1.weight', 'gate.proj.1.bias', 'gate.proj.3.weight', 'gate.proj.3.bias']:
    #     print(k, v.requires_grad)

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            # print(f"Parameter name: {name}, Shape: {param.shape}, Number of Parameters: {num_params}")
            print(name)
            total_params += num_params

    # print(f"Total Trainable Parameters: {total_params/1e6}")
    # from thop.profile import profile
    #
    # name = "our"
    # total_ops, total_params = profile(model, (HF,))
    # print(
    #     "%s         | %.4f(M)      | %.4f(G)         |"
    #     % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3))
    # )
