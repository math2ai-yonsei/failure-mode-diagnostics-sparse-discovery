#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate3 Trajectory VAE Model
==========================
Cart-Pole 4D 궤적 생성을 위한 VAE 모델.

Step6 패턴을 기반으로 하되, Cart-Pole 시스템에 맞게 적응.

Lock-3 준수: VAE는 x (상태)만 생성, dx는 생성하지 않음.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import warnings


@dataclass
class VAEConfig:
    """VAE 설정 (YAML에서 로드)"""
    # Architecture
    latent_dim: int = 8
    hidden_dim: int = 128
    state_dim: int = 4          # Cart-Pole: [x, x_dot, theta, theta_dot]
    seq_len: int = 100          # 시퀀스 길이 (데이터에서 추론)
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    
    # Beta schedule (KL annealing)
    beta_start: float = 0.01
    beta_end: float = 0.5
    warmup_epochs: int = 30
    
    # Generation
    n_generate: int = 100
    n_select: int = 10
    temperature: float = 1.0
    
    # Seeds
    vae_seed: int = 0
    gen_seed: int = 0
    
    @classmethod
    def from_yaml_dict(cls, config: Dict) -> 'VAEConfig':
        """YAML dict에서 VAEConfig 생성"""
        gen_cfg = config.get('generator', {})
        seeds_cfg = config.get('seeds', {})
        
        return cls(
            latent_dim=gen_cfg.get('latent_dim', 8),
            hidden_dim=gen_cfg.get('hidden_dim', 128),
            epochs=gen_cfg.get('epochs', 100),
            batch_size=gen_cfg.get('batch_size', 32),
            lr=gen_cfg.get('lr', 1e-3),
            beta_start=gen_cfg.get('beta_schedule', {}).get('start', 0.01),
            beta_end=gen_cfg.get('beta_schedule', {}).get('end', 0.5),
            warmup_epochs=gen_cfg.get('beta_schedule', {}).get('warmup_epochs', 30),
            n_generate=gen_cfg.get('n_generate', 100),
            n_select=gen_cfg.get('n_select', 10),
            temperature=gen_cfg.get('temperature', 1.0),
            vae_seed=seeds_cfg.get('vae', 0),
            gen_seed=seeds_cfg.get('gen', 0),
        )


class TrajectoryEncoder(nn.Module):
    """궤적을 latent space로 인코딩"""
    
    def __init__(self, state_dim: int, seq_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        
        # Flatten input: (batch, seq_len, state_dim) -> (batch, seq_len * state_dim)
        input_dim = seq_len * state_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, state_dim)
        Returns:
            mu, logvar: (batch, latent_dim)
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class TrajectoryDecoder(nn.Module):
    """Latent에서 궤적 복원"""
    
    def __init__(self, latent_dim: int, hidden_dim: int, seq_len: int, state_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim
        
        output_dim = seq_len * state_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_recon: (batch, seq_len, state_dim)
        """
        batch_size = z.shape[0]
        x_flat = self.decoder(z)
        x_recon = x_flat.view(batch_size, self.seq_len, self.state_dim)
        
        return x_recon


class TrajectoryVAE(nn.Module):
    """
    Cart-Pole 4D 궤적 VAE
    
    Lock-3 준수: x만 생성, dx는 외부에서 계산
    """
    
    def __init__(self, config: VAEConfig, seq_len: Optional[int] = None):
        super().__init__()
        self.config = config
        
        # seq_len은 데이터에서 추론하거나 명시적으로 제공
        self.seq_len = seq_len if seq_len is not None else config.seq_len
        self.state_dim = config.state_dim
        self.latent_dim = config.latent_dim
        
        # Encoder / Decoder
        self.encoder = TrajectoryEncoder(
            state_dim=self.state_dim,
            seq_len=self.seq_len,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim
        )
        
        self.decoder = TrajectoryDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            seq_len=self.seq_len,
            state_dim=self.state_dim
        )
        
        # Training state
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        
        # Data statistics (for denormalization)
        self.register_buffer('data_mean', torch.zeros(self.state_dim))
        self.register_buffer('data_std', torch.ones(self.state_dim))
    
    def set_data_statistics(self, train_x: np.ndarray):
        """학습 데이터 통계 저장 (정규화용)"""
        # train_x: (N, T, state_dim)
        mean = train_x.mean(axis=(0, 1))
        std = train_x.std(axis=(0, 1))
        std = np.where(std < 1e-8, 1.0, std)  # 0 방지
        
        self.data_mean = torch.tensor(mean, dtype=torch.float32)
        self.data_std = torch.tensor(std, dtype=torch.float32)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """입력 정규화"""
        return (x - self.data_mean) / self.data_std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """출력 역정규화"""
        return x * self.data_std + self.data_mean
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, 
                       temperature: float = 1.0) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * temperature
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, state_dim) - 원본 스케일
        Returns:
            x_recon: 복원된 궤적 (원본 스케일)
            mu: latent mean
            logvar: latent log variance
        """
        # Normalize
        x_norm = self.normalize(x)
        
        # Encode
        mu, logvar = self.encoder(x_norm)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon_norm = self.decoder(z)
        
        # Denormalize
        x_recon = self.denormalize(x_recon_norm)
        
        return x_recon, mu, logvar
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        VAE Loss 계산
        
        Args:
            x: 원본 궤적
            x_recon: 복원된 궤적
            mu, logvar: latent 분포 파라미터
            beta: KL 가중치
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
        }
    
    def get_beta(self, epoch: int) -> float:
        """Beta annealing schedule"""
        if epoch < self.config.warmup_epochs:
            # Linear warmup
            return self.config.beta_start + \
                   (self.config.beta_end - self.config.beta_start) * \
                   (epoch / self.config.warmup_epochs)
        return self.config.beta_end
    
    def fit(self, train_x: np.ndarray, val_x: Optional[np.ndarray] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        VAE 학습
        
        Args:
            train_x: (N_train, T, state_dim) 학습 데이터
            val_x: (N_val, T, state_dim) 검증 데이터 (선택)
            verbose: 로그 출력 여부
        
        Returns:
            학습 이력 dict
        """
        # Set seed
        torch.manual_seed(self.config.vae_seed)
        np.random.seed(self.config.vae_seed)
        
        # Update seq_len from data
        self.seq_len = train_x.shape[1]
        
        # Set data statistics
        self.set_data_statistics(train_x)
        
        # Reinitialize encoder/decoder with correct seq_len if needed
        if self.encoder.seq_len != self.seq_len:
            self.encoder = TrajectoryEncoder(
                state_dim=self.state_dim,
                seq_len=self.seq_len,
                hidden_dim=self.config.hidden_dim,
                latent_dim=self.config.latent_dim
            )
            self.decoder = TrajectoryDecoder(
                latent_dim=self.config.latent_dim,
                hidden_dim=self.config.hidden_dim,
                seq_len=self.seq_len,
                state_dim=self.state_dim
            )
        
        # Prepare data
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        self.to(device)
        
        train_tensor = torch.tensor(train_x, dtype=torch.float32, device=device)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        if val_x is not None:
            val_tensor = torch.tensor(val_x, dtype=torch.float32, device=device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        
        # Training loop
        self.train_loss_history = []
        self.val_loss_history = []
        
        for epoch in range(self.config.epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0
            
            beta = self.get_beta(epoch)
            
            for (batch_x,) in train_loader:
                optimizer.zero_grad()
                
                x_recon, mu, logvar = self.forward(batch_x)
                losses = self.compute_loss(batch_x, x_recon, mu, logvar, beta)
                
                losses['total'].backward()
                optimizer.step()
                
                epoch_loss += losses['total'].item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self.train_loss_history.append(avg_train_loss)
            
            # Validation
            if val_x is not None:
                self.eval()
                with torch.no_grad():
                    x_recon, mu, logvar = self.forward(val_tensor)
                    val_losses = self.compute_loss(val_tensor, x_recon, mu, logvar, beta)
                    self.val_loss_history.append(val_losses['total'].item())
            
            # Logging
            if verbose and (epoch + 1) % 20 == 0:
                val_str = f", Val: {self.val_loss_history[-1]:.4f}" if val_x is not None else ""
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, β: {beta:.3f}{val_str}")
        
        return {
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
        }
    
    @torch.no_grad()
    def sample(self, n_samples: int, temperature: float = 1.0,
               seed: Optional[int] = None) -> np.ndarray:
        """
        새로운 궤적 샘플링
        
        Lock-3 준수: x만 생성, dx는 반환하지 않음
        
        Args:
            n_samples: 생성할 샘플 수
            temperature: 샘플링 다양성 (1.0 = 표준)
            seed: 랜덤 시드 (None이면 config.gen_seed 사용)
        
        Returns:
            generated_x: (n_samples, seq_len, state_dim)
        """
        if seed is None:
            seed = self.config.gen_seed
        
        torch.manual_seed(seed)
        
        self.eval()
        device = next(self.parameters()).device
        
        # Sample from prior
        z = torch.randn(n_samples, self.latent_dim, device=device) * temperature
        
        # Decode
        x_norm = self.decoder(z)
        
        # Denormalize
        x = self.denormalize(x_norm)
        
        return x.cpu().numpy()
    
    @torch.no_grad()
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """입력 궤적 복원"""
        self.eval()
        device = next(self.parameters()).device
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        x_recon, _, _ = self.forward(x_tensor)
        
        return x_recon.cpu().numpy()
    
    @torch.no_grad()
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """궤적을 latent로 인코딩"""
        self.eval()
        device = next(self.parameters()).device
        
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        x_norm = self.normalize(x_tensor)
        mu, logvar = self.encoder(x_norm)
        
        return mu.cpu().numpy(), logvar.cpu().numpy()
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환 (aug_manifest용)"""
        return {
            'type': 'TrajectoryVAE',
            'latent_dim': self.latent_dim,
            'hidden_dim': self.config.hidden_dim,
            'state_dim': self.state_dim,
            'seq_len': self.seq_len,
            'epochs': self.config.epochs,
            'lr': self.config.lr,
            'beta_final': self.config.beta_end,
            'vae_seed': self.config.vae_seed,
            'gen_seed': self.config.gen_seed,
            'train_loss_final': self.train_loss_history[-1] if self.train_loss_history else None,
            'n_parameters': sum(p.numel() for p in self.parameters()),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_vae_from_config(config: Dict, seq_len: int) -> TrajectoryVAE:
    """YAML config에서 VAE 생성"""
    vae_config = VAEConfig.from_yaml_dict(config)
    vae_config.seq_len = seq_len
    return TrajectoryVAE(vae_config, seq_len=seq_len)