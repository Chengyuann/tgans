import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import spectral_norm


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


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=100, beta_schedule='linear'):
        super(GaussianDiffusion, self).__init__()
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = self.linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        return torch.linspace(scale * 0.0001, scale * 0.02, timesteps)

    def cosine_beta_schedule(self, timesteps):
        return torch.cos(torch.linspace(0, torch.pi, timesteps))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    def p_mean_variance(self, model, x_t, t):
        pred_noise = model(x_t)
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
                         self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance





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

        if add_final_conv:
            self.final_conv = nn.Conv2d(ndf * 16, nz, 4, 1, 0, bias=False)
        else:
            self.final_conv = None

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        if self.final_conv:
            latent = self.final_conv(out5)
        else:
            latent = out5

        return latent


class DCDecoder(nn.Module):
    def __init__(self, isize, nz, ngf, act, normalize):
        super(DCDecoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

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

        self.final_convtrans = nn.Sequential(
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            ActLayer('tanh')
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
        self.diffusion = GaussianDiffusion(timesteps=1000, beta_schedule='linear')

    def forward(self, data, outz=False, apply_diffusion=False):
        z = self.Encoder(data)
        timesteps = torch.randint(0, self.diffusion.timesteps, (z.size(0),), device=z.device).long()
        if outz:
            return z
        else:
            if apply_diffusion:
                z_noisy = self.diffusion.q_sample(z, timesteps)
                z = self.diffusion.predict_start_from_noise(z_noisy, timesteps, torch.zeros_like(z_noisy))
            recon = self.Decoder(z)
            return recon



class Discriminator(nn.Module):
    def __init__(self, param):
        super(Discriminator, self).__init__()
        ndf, isize = param['net']['ndf'], param['net']['isize']
        act, normalize = param['net']['act'][0], param['net']['normalize']['d']

        # Initialize the layers with spectral normalization
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, ndf, 4, 2, 1, bias=False)),
            NormLayer(normalize, ndf, isize // 2),
            ActLayer(act)
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            NormLayer(normalize, ndf * 2, isize // 4),
            ActLayer(act)
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            NormLayer(normalize, ndf * 4, isize // 8),
            ActLayer(act)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            NormLayer(normalize, ndf * 8, isize // 16),
            ActLayer(act)
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            NormLayer(normalize, ndf * 16, isize // 32),
            ActLayer(act)
        )

        # Final layer for feature extraction and classification
        self.feat_extract_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 16, ndf * 16, 4, 1, 0, bias=False, groups=ndf * 16)),
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
    def __init__(self, num_layer=3, mel_bins=128, win_len=2048, hop_len=313,embedding_dim=512):#344
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
        out = self.conv_extractor(x)
        out = self.conv_encoder(out)
        out = out.squeeze(0)  # 移除批次维度
        return out



