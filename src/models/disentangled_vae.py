import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


def _mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class MainEncoder(nn.Module):
    def __init__(self, seq_len, n_pitch, hidden=(512, 256), feat_dim=256):
        super().__init__()
        self.net = _mlp([seq_len * n_pitch, *hidden, feat_dim])

    def forward(self, x):                
        return self.net(x.flatten(1))        


class FactorEncoder(nn.Module):
    def __init__(self, in_dim, main_feat_dim, latent_dim, hidden=(256, 256)):
        super().__init__()
        self.stream = _mlp([in_dim, *hidden])    
        fused = hidden[-1] + main_feat_dim
        self.fc_mu = nn.Linear(fused, latent_dim)
        self.fc_lv = nn.Linear(fused, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, v, main_feat):                 
        h = self.relu(self.stream(v))
        h = torch.cat([h, main_feat], dim=1)    
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_lv(h), LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


class SplitLatentDecoder(nn.Module):
    def __init__(self, latent_dim, seq_len, n_pitch, hidden=(256, 512)):
        super().__init__()
        self.seq_len, self.n_pitch = seq_len, n_pitch
        self.net = _mlp([latent_dim, *hidden, seq_len * n_pitch])

    def forward(self, z_combined, return_logits=False):
        logits = self.net(z_combined).view(-1, self.seq_len, self.n_pitch)
        return logits if return_logits else torch.sigmoid(logits)


class DisentangledVAE(pl.LightningModule):
    def __init__(self, seq_len=256, n_pitch=32, pitch_lo=40,
                 latent_dim=64, lr=1e-3, beta=0.5, anneal_steps=10_000,
                 pos_weight=5.0, main_feat_dim=512,
                 main_hidden=(2048, 1024), factor_hidden=(512, 512),
                 dec_hidden=(1024, 2048)):
        super().__init__()
        self.save_hyperparameters()

        self.main_encoder = MainEncoder(seq_len, n_pitch, main_hidden, main_feat_dim)
        self.encoder_rhythm = FactorEncoder(seq_len,  main_feat_dim, latent_dim, factor_hidden)
        self.encoder_pitch  = FactorEncoder(n_pitch,  main_feat_dim, latent_dim, factor_hidden)
        self.decoder = SplitLatentDecoder(latent_dim, seq_len, n_pitch, dec_hidden)

    @staticmethod
    def make_vr(x):                
        return x.sum(dim=2)

    @staticmethod
    def make_vp(x):          
        return x.sum(dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x):
        feat = self.main_encoder(x)
        mu_r, lv_r = self.encoder_rhythm(self.make_vr(x), feat)
        mu_p, lv_p = self.encoder_pitch(self.make_vp(x), feat)
        return mu_r, lv_r, mu_p, lv_p

    def forward(self, x):
        mu_r, lv_r, mu_p, lv_p = self.encode(x)
        z_r = self.reparameterize(mu_r, lv_r)
        z_p = self.reparameterize(mu_p, lv_p)
        recon = self.decoder(z_r + z_p)     
        return recon, mu_r, lv_r, mu_p, lv_p

    def _step(self, batch, stage):

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.float()

        mu_r, lv_r, mu_p, lv_p = self.encode(x)
        z_r = self.reparameterize(mu_r, lv_r)
        z_p = self.reparameterize(mu_p, lv_p)
        recon_logits = self.decoder(z_r + z_p, return_logits=True)

        B = x.size(0)
        
        pw = torch.tensor(self.hparams.pos_weight, device=x.device)
        recon_loss = F.binary_cross_entropy_with_logits(
            recon_logits.reshape(B, -1), x.reshape(B, -1),
            pos_weight=pw, reduction="sum")

        kl_r = -0.5 * torch.sum(1 + lv_r - mu_r.pow(2) - lv_r.exp())
        kl_p = -0.5 * torch.sum(1 + lv_p - mu_p.pow(2) - lv_p.exp())

        anneal = min(1.0, self.global_step / max(1, self.hparams.anneal_steps))
        loss = (recon_loss + anneal * self.hparams.beta * (kl_r + kl_p)) / B

        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_recon": recon_loss / B,
             f"{stage}_kl_r": kl_r / B,
             f"{stage}_kl_p": kl_p / B,
             "z_std_r": mu_r.std(), "z_std_p": mu_p.std(), "anneal": anneal},
            prog_bar=True, on_step=(stage == "train"), on_epoch=(stage != "train"),
        )
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)