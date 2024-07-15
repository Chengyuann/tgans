import torch
import torch.nn as nn

class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=2048, hop_len=344):  # 增加win_len值 #344 #313
        super(TgramNet, self).__init__()
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(512),  # 修改 LayerNorm 的输入维度
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, padding=2**idx, dilation=2**idx, bias=False),  # 使用扩张卷积
            ) for idx in range(num_layer)]
        )

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        out = self.conv_extractor(x)
        print(f"After conv_extractor shape: {out.shape}")
        out = self.conv_encoder(out)
        print(f"After conv_encoder shape: {out.shape}")
        out = out.squeeze(0)  # 移除批次维度
        print(f"Output shape: {out.shape}")
        return out

if __name__ == "__main__":
    # 创建一个随机的输入数据
    wav = torch.randn(1, 1, 176000) #160000
    print(f"Original wav shape: {wav.shape}")

    # 创建 TgramNet 模型实例
    tgram_net = TgramNet()

    # 传递输入数据并打印各层的输出维度
    tgram_feat = tgram_net(wav)
    tgram_feat = tgram_feat.mean(dim=0).detach().cpu().numpy()
    print(f"Output shape: {tgram_feat.shape}")
