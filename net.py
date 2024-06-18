import torch.nn as nn
import torch

def ActLayer(act):
    assert act in ['relu', 'leakyrelu', 'tanh'], 'Unknown activate function!'
    if act == 'relu':
        return nn.ReLU(True)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.2, True)
    elif act == 'tanh':
        return nn.Tanh()


def NormLayer(normalize, chan, reso):
    assert normalize in ['bn', 'ln', 'in'], 'Unknown normalize function!'
    if normalize == 'bn':
        return nn.BatchNorm2d(chan)
    elif normalize == 'ln':
        return nn.LayerNorm((chan, reso, reso))
    elif normalize == 'in':
        return nn.InstanceNorm2d(chan)




class DCEncoder(nn.Module):
    """
    DCGAN DCEncoder NETWORK
    """

    def __init__(self, isize, nz, ndf, act, normalize, add_final_conv=True):
        super(DCEncoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = []
        main.append(nn.Conv2d(1, ndf, 4, 2, 1, bias=False))
        main.append(NormLayer(normalize, ndf, isize // 2))
        main.append(ActLayer(act))
        csize, cndf = isize // 2, ndf
        ##这一序列从一个卷积层开始，该层将输入尺寸减半（步长为2）并将深度增加到 ndf。接着添加一个指定类型的归一化层。然后添加一个指定类型的激活层。csize 和 cndf 用来跟踪特征图的当前尺寸和深度。
        
        while csize > 4:
            in_chan = cndf
            out_chan = cndf * 2
            main.append(nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False))
            cndf = cndf * 2
            csize = csize // 2
            main.append(NormLayer(normalize, out_chan, csize))
            main.append(ActLayer(act))
        ##这个循环不断添加卷积层、归一化层和激活层的组合。每个卷积层继续将特征图尺寸减半（csize）并将深度加倍（cndf），这一模式持续到 csize 减小到4为止。

        # state size. K x 4 x 4
        if add_final_conv:
            main.append(nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        #如果 add_final_conv 为真，则添加一个额外的卷积层，该层不改变空间尺寸但将通道深度调整为 nz，即潜在维度的数量。
        self.main = nn.Sequential(*main)

    def forward(self, x):
        z = self.main(x)
        return z


class DCDecoder(nn.Module):
    """
    DCGAN DCDecoder NETWORK
    """
    def __init__(self, isize, nz, ngf, act, normalize):
        super(DCDecoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        ##计算初始的特征图深度和大小。从一个小尺寸（4x4）开始，逐渐放大至目标尺寸（isize）。每次循环将特征图尺寸翻倍，同时将通道数增加一倍。


        main = []
        main.append(nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        csize = 4
        main.append(NormLayer(normalize, cngf, csize))
        main.append(ActLayer(act))
        ##首先添加一个转置卷积层，从潜在空间开始生成图像。这一层将特征图的尺寸从1x1变到4x4。紧接着添加归一化层和激活层。

        while csize < isize // 2:
            main.append(nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            cngf = cngf // 2
            csize = csize * 2
            main.append(NormLayer(normalize, cngf, csize))
            main.append(ActLayer(act))
        #这个循环逐步添加转置卷积层，每层都将特征图的尺寸扩大一倍，并将通道数减半，直到达到目标尺寸的一半为止。

        main.append(nn.ConvTranspose2d(cngf, 1, 4, 2, 1, bias=False))
        main.append(ActLayer('tanh'))
        self.main = nn.Sequential(*main)
        #最后一层转置卷积将特征图的尺寸扩大到最终的图像尺寸（isize x isize），通道数减少到1，这意味着输出是单通道图像（比如灰度图）。使用 'tanh' 激活函数，它通常用于将输出归一化到[-1, 1]，这对生成图像特别有效。
    
    def forward(self, z):
        x = self.main(z)
        return x


class AEDC(nn.Module):
    def __init__(self, param):
        super(AEDC, self).__init__()
        self.Encoder = DCEncoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ndf=param['net']['ndf'],
                                 act=param['net']['act'][0],
                                 normalize=param['net']['normalize']['g'])
        self.Decoder = DCDecoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ngf=param['net']['ngf'],
                                 act=param['net']['act'][1],
                                 normalize=param['net']['normalize']['g'])

    def forward(self, data, outz=False):
        z = self.Encoder(data)
        if outz:
            return z
        else:
            recon = self.Decoder(z)
            return recon
        ##如果 outz 为 True，返回潜在向量 z。如果 outz 为 False，则将 z 传递给解码器生成重构图像 recon 并返回。

class Discriminator(nn.Module):
    def __init__(self, param):
        super(Discriminator, self).__init__()
        ndf, isize = param['net']['ndf'], param['net']['isize']
        act, normalize = param['net']['act'][0], param['net']['normalize']['d']

        self.main = nn.ModuleList()
        level = 0
        in_chan = 1
        chans, resoes = [in_chan], [isize]
        init_layer = nn.Sequential(nn.Conv2d(in_chan, ndf, 4, 2, 1, bias=False),
                                   NormLayer(normalize, ndf, isize // 2),
                                   ActLayer(act))
        level, csize, cndf = 1, isize // 2, ndf
        self.main.append(init_layer)
        chans.append(ndf)
        resoes.append(csize)

        while csize > 4:
            in_chan = cndf
            out_chan = cndf * 2
            pyramid = [nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False)]
            level, cndf, csize = level + 1, cndf * 2, csize // 2
            pyramid.append(NormLayer(normalize, out_chan, csize))
            pyramid.append(ActLayer(act))
            self.main.append(nn.Sequential(*pyramid))
            chans.append(out_chan)
            resoes.append(csize)

        in_chan = cndf
        # 判断真假
        self.feat_extract_layer = nn.Sequential(nn.Conv2d(in_chan, in_chan, 4, 1, 0, bias=False, groups=in_chan),  # GDConv
                                                nn.Flatten())  # D网络的embedding
        self.output_layer = nn.Sequential(nn.LayerNorm(in_chan),
                                          ActLayer(act),
                                          nn.Linear(in_chan, 1))

    def forward(self, x):
        for module in self.main:
            x = module(x)
        feat = self.feat_extract_layer(x)
        pred = self.output_layer(feat)
        return pred, feat

class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out