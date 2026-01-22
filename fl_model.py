
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TinyMLModel(nn.Module):
    """
    Improved lightweight model for Edge-IIoT dataset
    Ensures all operations use float tensors
    """
    def __init__(self, input_dim=62, hidden_dim=128, output_dim=15, dropout_rate=0.3):
        super().__init__()
        
        # Deeper architecture with better capacity
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Ensure all parameters are float32
        self.to(torch.float32)
        
        # Calculate model size
        self.model_size = self._calculate_model_size()
        
    def _initialize_weights(self):
        """Xavier/He initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight.float())
                if m.bias is not None:
                    nn.init.constant_(m.bias.float(), 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.float(), 1)
                nn.init.constant_(m.bias.float(), 0)
    
    def _calculate_model_size(self):
        """Calculate model size in KB"""
        total_params = sum(p.numel() for p in self.parameters())
        size_kb = (total_params * 4) / 1024
        return size_kb
    
    def forward(self, x):
        # Ensure input is float
        if x.dtype != torch.float32:
            x = x.float()
            
        # Layer 1
        x = self.fc1(x)
        if x.shape[0] > 1:  # BatchNorm needs batch_size > 1
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x
    
    def get_model_info(self):
        """Get model information for logging"""
        return {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "model_size_kb": self.model_size,
            "layers": len(list(self.modules()))
        }


class SimpleMLP(nn.Module):
    """
    Very simple MLP for debugging - ensures float tensors
    """
    def __init__(self, input_dim=62, hidden_dim=64, output_dim=15):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Simple initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.float())
                nn.init.xavier_uniform_(m.weight.float(), gain=0.1)  # Add gain=0.1
                nn.init.zeros_(m.bias.float())
        
        # Ensure float32
        self.to(torch.float32)
    
    def forward(self, x):
        # Ensure input is float
        if x.dtype != torch.float32:
            x = x.float()
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AnomalyDetector(nn.Module):
    """
    Lightweight Autoencoder for N-BaIoT anomaly detection
    Uses reconstruction error for anomaly detection
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[32, 16]):
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        
        for i in range(len(hidden_dims)-1, 0, -1):
            decoder_layers.extend([
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.Linear(hidden_dims[i], hidden_dims[i-1])
            ])
        
        decoder_layers.extend([
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], input_dim)
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Ensure float32
        self.to(torch.float32)
        
        # Calculate model size
        self.model_size = self._calculate_model_size()
        
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.float())
                if m.bias is not None:
                    nn.init.constant_(m.bias.float(), 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.float(), 1)
                nn.init.constant_(m.bias.float(), 0)
    
    def _calculate_model_size(self):
        """Calculate model size in KB"""
        total_params = sum(p.numel() for p in self.parameters())
        size_kb = (total_params * 4) / 1024
        return size_kb
    
    def forward(self, x):
        # Ensure input is float
        if x.dtype != torch.float32:
            x = x.float()
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Compute reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = F.mse_loss(reconstructed, x, reduction='none')
            error = torch.mean(error, dim=1)
        return error
    
    def get_model_info(self):
        """Get model information"""
        return {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "model_size_kb": self.model_size,
            "layers": len(list(self.modules()))
        }


def get_model(config, input_dim, num_classes=None):
    """
    Get model based on dataset and configuration
    """
    
    if config.dataset == "nbaiot":
        model = AnomalyDetector(
            input_dim=input_dim,
            latent_dim=8,
            hidden_dims=[32, 16]
        )
        print(f"Initialized AnomalyDetector: {model.get_model_info()}")
        
    elif config.dataset == "edge_iiot":
        # Use SimpleMLP for better stability
        model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=num_classes or 15
        )
        print(f"Initialized SimpleMLP for stability")
        
    else:
        # Default to simple model
        model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=num_classes or 2
        )
    
    # Ensure model uses float32
    model = model.float()
    
    # Check model parameters are float
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"Warning: {name} has dtype {param.dtype}, converting to float32")
            param.data = param.data.float()
    
    return model


def test_model_robustness(model, test_input):
    """
    Test model for numerical stability
    """
    model.eval()
    results = {}
    
    # Ensure test input is float
    if test_input.dtype != torch.float32:
        test_input = test_input.float()
    
    with torch.no_grad():
        # Test normal input
        output = model(test_input)
        results['normal_output_mean'] = output.mean().item()
        results['normal_output_std'] = output.std().item()
        
        # Test with zero input
        zero_input = torch.zeros_like(test_input).float()
        zero_output = model(zero_input)
        results['zero_input_stable'] = not torch.isnan(zero_output).any().item()
        
        # Test with large input
        large_input = test_input * 100
        large_output = model(large_input)
        results['large_input_stable'] = not torch.isnan(large_output).any().item()
        
        # Test gradient flow
        model.train()
        test_input.requires_grad = True
        output = model(test_input)
        loss = output.sum()
        loss.backward()
        
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        results['gradient_flow_healthy'] = all(g < 100 for g in grad_norms)
        results['max_gradient_norm'] = max(grad_norms) if grad_norms else 0
    
    return results





if __name__ == "__main__":
    # Test SimpleMLP model
    print("Testing SimpleMLP Model for Edge-IIoT:")
    model = SimpleMLP(input_dim=62, hidden_dim=64, output_dim=15)
    
    # Check all parameters are float
    for name, param in model.named_parameters():
        print(f"{name}: dtype={param.dtype}, shape={param.shape}")
    
    # Test with sample data
    batch_size = 32
    test_input = torch.randn(batch_size, 62).float()  # Ensure float
    output = model(test_input)
    print(f"\nOutput shape: {output.shape}, dtype: {output.dtype}")
    
    # Test robustness
    robustness = test_model_robustness(model, test_input)
    print(f"Robustness tests: {robustness}")