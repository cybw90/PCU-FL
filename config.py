
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Config:
    """PCU-FL Configuration - Matching previous high-performance setup"""
    
    # Dataset
    dataset: str = "edge_iiot"
    
    # FL System Parameters
    num_clients: int = 100
    clients_per_round: int = 30
    num_rounds: int = 100
    local_epochs: int = 1  
    batch_size: int = 32
    learning_rate: float = 0.1
    
    # Privacy Parameters
    clip_norm: float = 0.2  # Start at 1.0, let adaptive adjust down
    l_inf_bound: float = 1.0
    ring_dim: int = 4096
    epsilon_per_round: float = 1.0
    delta: float = 1e-5
    W_min: float = 1.0
    
    # Integer mode for Algorithm 3
    dp_integer_mode: bool = True
    
    # Quantization parameters
    fp8_mode: str = "none"
    gamma: int = 4096
    alpha: int = 256
    

    # ENSURE these are set:
    eta_max: float = 0.0
    initial_budget: int = 100
    threat_threshold: float = 0.3
    
    # Server operations
    server_lr: float = 1.0
    server_update_clip: float = 10.0  # Reasonable trust region
    use_momentum: bool = False
    ema_decay: float = 0.99  # Use EMA for evaluation
    
    # Adaptive clipping
    adaptive_clipping: bool = True
    target_clip_rate: float = 0.2
    histogram_epsilon: float = 5.0
    histogram_delta: float = 1e-7
    histogram_bins: int = 50
    quantile_target: float = 0.8
    safety_factor: float = 1.0
    histogram_mechanism: str = "laplace"
    
    # Trust Budget
    initial_budget: int = 100
    threat_threshold: float = 0.3
    
    # RLWE Parameters
    # modulus_t: int = 2**32 - 5
    modulus_t: int = 2**31 - 1  # Standard prime for RLWE
    packing_kappa: int = 1
    
    # Evaluation settings
    positive_class_index: int = 1
    auto_flip_auc: bool = False
    temperature_calibration: bool = True
    
    # Wrap probability
    delta_wrap: float = 1e-12
    tau_wrap: float = None
    
    # Training settings
    warmup_rounds: int = 5
    patience: int = 40
    task_mode: str = "binary_dominant"
    data_distribution: str = "iid"
    
    # Additional parameters
    w_max: float = 1.0
    eta_max: float = 0.0
    strict_proof_verification: bool = False
    log_proof_diag: bool = False
    verifiable_noise: bool = False
    
    def __post_init__(self):
        """Minimal post-init - don't override critical values"""
        # Validate participation
        self.clients_per_round = min(self.clients_per_round, self.num_clients)
        
        # Set quantization scales if FP8 is used
        if self.fp8_mode == "E4M3":
            self.gamma = 256
            self.alpha = 256
        elif self.fp8_mode == "E5M2":
            self.gamma = 512
            self.alpha = 512
        
        # Compute tau for wrap probability
        if self.tau_wrap is None:
            try:
                from scipy.stats import norm
                L = self.ring_dim * self.packing_kappa
                self.tau_wrap = norm.ppf(1 - self.delta_wrap / (2 * L))
            except ImportError:
                self.tau_wrap = 6.0
        
        # Print configuration
        print(f"\nPCU-FL Configuration:")
        print(f"  Privacy: ε={self.epsilon_per_round}/round, δ={self.delta}")
        print(f"  Clients: {self.clients_per_round}/{self.num_clients}")
        print(f"  Training: {self.local_epochs} epochs, batch={self.batch_size}")
        print(f"  Clipping: S={self.clip_norm}, adaptive={self.adaptive_clipping}, target={self.target_clip_rate}")
        print(f"  Server: lr={self.server_lr}, trust_region={self.server_update_clip}")
        print(f"  EMA: {self.ema_decay}")