import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CBAM(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_weight = self.channel_att(x)
        x_channel = x * channel_weight

        spatial_avg = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_feat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_weight = self.spatial_att(spatial_feat)
        x_final = x_channel * spatial_weight

        return x_final


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.randn(1, channel // (2 * G), 1, 1) * 0.02)
        self.cbias = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.randn(1, channel // (2 * G), 1, 1) * 0.02)
        self.sbias = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)

        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out


class ParallelAttention(nn.Module):

    def __init__(self, channels, reduction=16, G=8):
        super().__init__()
        self.shuffle_att = ShuffleAttention(channels, G=G)
        self.cbam = CBAM(channels, reduction=reduction)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        shuffle_out = self.shuffle_att(x)
        cbam_out = self.cbam(x)

        total_weight = torch.abs(self.alpha) + torch.abs(self.beta)
        alpha_norm = torch.abs(self.alpha) / total_weight
        beta_norm = torch.abs(self.beta) / total_weight

        fused_output = alpha_norm * shuffle_out + beta_norm * cbam_out

        return fused_output


class MobileNetV1_ShuffleAtt(nn.Module):
    def __init__(self, num_classes=3, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

        # 基础卷积块
        def conv_bn(in_p, out_p, stride):
            return nn.Sequential(
                nn.Conv2d(in_p, out_p, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_p),
                nn.ReLU(inplace=True))

        # 深度可分离卷积块
        def conv_dw(in_p, out_p, stride):
            return nn.Sequential(
                nn.Conv2d(in_p, in_p, 3, stride, 1, groups=in_p, bias=False),
                nn.BatchNorm2d(in_p),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_p, out_p, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_p),
                nn.ReLU(inplace=True))

        # 使用并联注意力的卷积块
        def conv_dw_parallel_att(in_p, out_p, stride):
            return nn.Sequential(
                conv_dw(in_p, out_p, stride),
                ParallelAttention(out_p, reduction=16, G=8)  # 并联注意力
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),  # 112x112
            conv_dw_parallel_att(32, 64, 1),  # 112x112
            conv_dw_parallel_att(64, 128, 2),  # 56x56
            conv_dw_parallel_att(128, 128, 1),  # 56x56
            conv_dw_parallel_att(128, 256, 2),  # 28x28
            conv_dw_parallel_att(256, 256, 1),  # 28x28
            conv_dw_parallel_att(256, 512, 2),  # 14x14
            # 前3层使用并联注意力
            conv_dw_parallel_att(512, 512, 1),
            conv_dw_parallel_att(512, 512, 1),
            conv_dw_parallel_att(512, 512, 1),
            # 后2层只用深度可分离卷积
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw_parallel_att(512, 1024, 2),  # 7x7
            conv_dw(1024, 1024, 1),  # 最后一层不用注意力
            nn.AdaptiveAvgPool2d(1)  # 1x1
        )

        self.fc = nn.Linear(1024, num_classes)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    def forward(self, x, target=None):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # 展平特征
        x = self.fc(x)

        if target is not None:
            loss = self.criterion(x, target)
            return x, loss
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = MobileNetV1_ShuffleAtt(num_classes=3, label_smoothing=0.1)
    model = model.to(device)
    print(f"模型设备: {next(model.parameters()).device}")

    # 测试前向传播
    print("\n🔍 测试模型功能...")
    x = torch.randn(2, 3, 224, 224).to(device)
    target = torch.randint(0, 3, (2,)).to(device)

    # 测试不带target
    output = model(x)
    print(f"不带target输出形状: {output.shape}")

    # 测试带target
    output, loss = model(x, target)
    print(f"带target输出形状: {output.shape}, 损失值: {loss.item():.4f}")

    # 特别检查并联注意力的权重
    print("\n🔍 并联注意力权重:")
    for name, param in model.named_parameters():
        if 'alpha' in name or 'beta' in name:
            print(f"{name:50} value: {param.data.item():.4f}")