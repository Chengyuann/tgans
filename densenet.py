import torch.nn as nn
import torch
import torch.nn.functional as F

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
    def __init__(self, isize, nz, ndf, act, normalize, add_final_conv=True):
        super(DCEncoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf, isize // 2),
            ActLayer(act)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf + ndf, ndf * 2, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 2, isize // 4),
            ActLayer(act)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf + ndf * 2 + ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 4, isize // 8),
            ActLayer(act)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf + ndf * 2 + ndf * 4 + ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 8, isize // 16),
            ActLayer(act)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 16, isize // 32),
            ActLayer(act)
        )

        if add_final_conv:
            self.final_conv = nn.Conv2d(ndf * 16, nz, 4, 1, 0, bias=False)
        else:
            self.final_conv = None

    def forward(self, x):
        out1 = self.conv1(x)
        out2_input = torch.cat([out1, out1], 1)
        out2 = self.conv2(out2_input)

        out1_resized = F.interpolate(out1, size=out2.size()[2:], mode='nearest')
        out2_resized = F.interpolate(out2, size=out2.size()[2:], mode='nearest')
        out3_input = torch.cat([out1_resized, out2_resized, out2_resized], 1)
        out3 = self.conv3(out3_input)

        out1_resized = F.interpolate(out1, size=out3.size()[2:], mode='nearest')
        out2_resized = F.interpolate(out2, size=out3.size()[2:], mode='nearest')
        out3_resized = F.interpolate(out3, size=out3.size()[2:], mode='nearest')
        out4_input = torch.cat([out1_resized, out2_resized, out3_resized, out3_resized], 1)
        out4 = self.conv4(out4_input)

        out5 = self.conv5(out4)

        if self.final_conv:
            x = self.final_conv(out5)
        else:
            x = out5
        return x

class DCDecoder(nn.Module):
    """
    DCGAN DCDecoder NETWORK
    """
    def __init__(self, isize, nz, ngf, act, normalize):
        super(DCDecoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # Initialize the transposed convolutional layers
        self.convtrans1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            NormLayer(normalize, ngf * 16, 4),
            ActLayer(act)
        )

        self.convtrans2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            NormLayer(normalize, ngf * 8, 8),
            ActLayer(act)
        )

        self.convtrans3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            NormLayer(normalize, ngf * 4, 16),
            ActLayer(act)
        )

        self.convtrans4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            NormLayer(normalize, ngf * 2, 32),
            ActLayer(act)
        )

        self.convtrans5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            NormLayer(normalize, ngf, 64),
            ActLayer(act)
        )

        # Optional final convolutional layer to match the output size and channels
        self.final_convtrans = nn.Sequential(
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            ActLayer('tanh')  # 添加激活层以确保输出归一化
        )

    def forward(self, z):
        x = self.convtrans1(z)
        x = self.convtrans2(x)
        x = self.convtrans3(x)
        x = self.convtrans4(x)
        x = self.convtrans5(x)
        x = self.final_convtrans(x)
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

        # Initialize the layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf, isize // 2),
            ActLayer(act)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 2, isize // 4),
            ActLayer(act)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 4, isize // 8),
            ActLayer(act)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 8, isize // 16),
            ActLayer(act)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf * 16, isize // 32),
            ActLayer(act)
        )

        # Final layer for feature extraction and classification
        self.feat_extract_layer = nn.Sequential(
            nn.Conv2d(ndf * 16, ndf * 16, 4, 1, 0, bias=False, groups=ndf * 16),
            nn.Flatten()
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(ndf * 16),
            ActLayer(act),
            nn.Linear(ndf * 16, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        feat = self.feat_extract_layer(x)
        pred = self.output_layer(feat)
        return pred, feat
    




class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=313, embedding_dim=512):
        super(TgramNet, self).__init__()
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)]
        )


    def forward(self, x):
        out = self.conv_extractor(x)
        out = self.conv_encoder(out)
        out = out.squeeze(0) 
        return out
