import torch
import torch.nn as nn

class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512, embedding_dim=512):
        super(TgramNet, self).__init__()
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)]
        )
        
        # 计算卷积层输出的形状，并调整全连接层的输入形状
        self.conv_output_size = self._get_conv_output_size()

        self.fc = nn.Linear(self.conv_output_size, embedding_dim)  # 确保输出维度匹配

    def _get_conv_output_size(self):
        dummy_input = torch.randn(1, 1, 160000)  # 假设输入为 (1, 1, 160000)
        dummy_output = self.conv_extractor(dummy_input)
        dummy_output = self.conv_encoder(dummy_output)
        return dummy_output.view(1, -1).size(1)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        out = self.conv_extractor(x)
        print(f"After conv_extractor shape: {out.shape}")
        out = self.conv_encoder(out)
        print(f"After conv_encoder shape: {out.shape}")
        out = out.view(out.size(0), -1)  # 展平输出
        print(f"After flattening shape: {out.shape}")
        out = self.fc(out)  # 应用全连接层
        print(f"After fully connected shape: {out.shape}")
        return out

if __name__ == "__main__":
    # 创建一个随机的输入数据
    wav = torch.randn(1, 1, 160000)
    print(f"Original wav shape: {wav.shape}")

    # 创建 TgramNet 模型实例
    tgram_net = TgramNet()

    # 传递输入数据并打印各层的输出维度
    tgram_feat = tgram_net(wav)
    print(f"Output shape: {tgram_feat.shape}")
