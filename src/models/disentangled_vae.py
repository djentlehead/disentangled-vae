import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, seq_len=256, latent_dim=128, channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(4, 12), stride=(2, 2), padding=(1, 5)) # -> (B, C, 128, 64)
        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=(4, 12), stride=(2, 2), padding=(1, 5)) # -> (B, C*2, 64, 32)
        self.conv3 = nn.Conv2d(channels*2, channels*4, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3)) # -> (B, C*4, 32, 16)
        self.conv4 = nn.Conv2d(channels*4, channels*8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)) # -> (B, C*8, 16, 8)
        
        self.flat_size = channels * 8 * 16 * 8
        
        self.fc = nn.Linear(self.flat_size, 1024)

        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc(x))
        
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        return mean, log_var

class Decoder(nn.Module):

    def __init__(self, seq_len=256, latent_dim=128, channels=32):
        super().__init__()
        
        self.combined_latent_dim = latent_dim

        self.init_channels = channels * 8
        self.init_dims = (16, 8) 
        self.flat_size = self.init_channels * self.init_dims[0] * self.init_dims[1]

        self.fc1 = nn.Linear(self.combined_latent_dim, 1024)
        self.fc2 = nn.Linear(1024, self.flat_size)
        
        self.deconv1 = nn.ConvTranspose2d(self.init_channels, channels*4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)) # -> (B, C*4, 32, 16)
        self.deconv2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3)) # -> (B, C*2, 64, 32)
        self.deconv3 = nn.ConvTranspose2d(channels*2, channels, kernel_size=(4, 12), stride=(2, 2), padding=(1, 5)) # -> (B, C, 128, 64)
        self.deconv4 = nn.ConvTranspose2d(channels, 1, kernel_size=(4, 12), stride=(2, 2), padding=(1, 5)) # -> (B, 1, 256, 128)

        self.relu = nn.ReLU()

    def forward(self, z_rhythm, z_pitch):
        z_combined = torch.cat([z_rhythm, z_pitch], dim=1)
        
        x = self.relu(self.fc1(z_combined))
        x = self.relu(self.fc2(x))
        
        x = x.view(x.size(0), self.init_channels, self.init_dims[0], self.init_dims[1])
        
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        
        x = torch.sigmoid(self.deconv4(x))
        
        x = x.squeeze(1)
        
        return x

class DisentangledVAE(pl.LightningModule):
    def __init__(self, seq_len=256, latent_dim_rhythm=64, latent_dim_pitch=64, 
                 lr=1e-3, beta_r=1.0, beta_p=1.0, channels=32):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_rhythm = Encoder(seq_len=seq_len, latent_dim=latent_dim_rhythm, channels=channels)
        self.encoder_pitch = Encoder(seq_len=seq_len, latent_dim=latent_dim_pitch, channels=channels)
        
        self.decoder = Decoder(seq_len=seq_len, latent_dim=latent_dim_rhythm + latent_dim_pitch, channels=channels)
        
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x_rhythm, x_pitch):
        mean_r, log_var_r = self.encoder_rhythm(x_rhythm)
        mean_p, log_var_p = self.encoder_pitch(x_pitch)
        
        z_r = self.reparameterize(mean_r, log_var_r)
        z_p = self.reparameterize(mean_p, log_var_p)
        
        x_recon = self.decoder(z_r, z_p)
        return x_recon, mean_r, log_var_r, mean_p, log_var_p

    def training_step(self, batch, batch_idx):
        x_original, x_rhythm, x_pitch = batch
        
        mean_r, log_var_r = self.encoder_rhythm(x_rhythm)
        
        mean_p, log_var_p = self.encoder_pitch(x_pitch)
        
        z_r = self.reparameterize(mean_r, log_var_r)
        z_p = self.reparameterize(mean_p, log_var_p)
        
        x_reconstructed = self.decoder(z_r, z_p)
        
        loss_recon = F.binary_cross_entropy(
            x_reconstructed.view(x_reconstructed.size(0), -1), 
            x_original.view(x_original.size(0), -1), 
            reduction='sum'
        )
        
        loss_kl_r = -0.5 * torch.sum(1 + log_var_r - mean_r.pow(2) - log_var_r.exp())
        
        loss_kl_p = -0.5 * torch.sum(1 + log_var_p - mean_p.pow(2) - log_var_p.exp())
        
        batch_size = x_original.size(0)
        total_loss = (
            (loss_recon + 
             self.hparams.beta_r * loss_kl_r + 
             self.hparams.beta_p * loss_kl_p) / batch_size
        )
        
        self.log_dict({
            'train_loss': total_loss,
            'loss_recon': loss_recon / batch_size,
            'loss_kl_rhythm': loss_kl_r / batch_size,
            'loss_kl_pitch': loss_kl_p / batch_size
        }, prog_bar=True, on_step=True, on_epoch=False)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer