
import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict
import copy

class ServerOps:
    """Enhanced server-side operations for PCU-FL"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        # self.trust_budgets = defaultdict(lambda: config.initial_budget)
        self.trust_budgets = defaultdict(lambda: config.initial_budget)
        # self.trust_budgets = defaultdict(lambda: 2) 
        self.round_num = 0
        self.momentum_buffer = {}
        self.best_accuracy = 0.0
        self.best_model_state = None
        
    def verify_and_gate(self, client_submissions: List[Tuple[int, bytes, bytes]]) -> List[int]:
        """Accept clients based on trust budget"""
        accepted_clients = []
        
        for client_id, ciphertext, proof in client_submissions:
            # Simple trust-based gating
            if self.trust_budgets[client_id] > 0:
                accepted_clients.append(client_id)
            else:
                # Restore budget after cooldown
                if self.round_num % 10 == 0:
                    self.trust_budgets[client_id] = self.config.initial_budget // 2
                
        return accepted_clients
    
    def aggregate_with_dp(self, 
                         client_updates: Dict[int, Dict], 
                         accepted_clients: List[int]) -> Dict:
        """Improved aggregation with calibrated DP noise"""
        
        if not accepted_clients or not client_updates:
            return {}
        
        first_update = next(iter(client_updates.values()))
        aggregated = {}
        n_clients = len(accepted_clients)
        
        for key in first_update.keys():
            updates_list = []
            for client_id in accepted_clients:
                if client_id in client_updates and key in client_updates[client_id]:
                    update = client_updates[client_id][key].float()
                    if not torch.isnan(update).any() and not torch.isinf(update).any():
                        updates_list.append(update)
            
            if not updates_list:
                continue
                
            # Compute average
            stacked = torch.stack(updates_list)
            avg_update = torch.mean(stacked, dim=0)
            
            if self.config.epsilon_per_round <= 0.5:
                # Very strong privacy: use variance-based noise
                if len(updates_list) > 1:
                    update_variance = torch.var(stacked, dim=0).mean().item()
                    noise_scale = np.sqrt(update_variance) * 0.05  # 5% of std dev
                else:
                    noise_scale = 1e-6
                    
                if 'bias' in key:
                    noise_scale *= 0.1
                    
            elif self.config.epsilon_per_round <= 1.0:
               
                if len(updates_list) > 1:
                   
                    update_variance = torch.var(stacked, dim=0).mean().item()
                    signal_magnitude = torch.abs(avg_update).mean().item()
                    
                    
                    if update_variance > 0:
                        noise_scale = np.sqrt(update_variance) * 0.02  # 2% of std dev
                    elif signal_magnitude > 0:
                        noise_scale = 0.0005 * signal_magnitude  # 0.05% of signal
                    else:
                        noise_scale = 1e-6
                else:
                    signal_magnitude = torch.abs(avg_update).mean().item()
                    noise_scale = 0.0005 * signal_magnitude if signal_magnitude > 0 else 1e-6
                
                if 'bias' in key:
                    noise_scale *= 0.1
                    
            elif self.config.epsilon_per_round < 10:
                # Moderate privacy
                sensitivity = self.config.clip_norm / n_clients
                sigma = self._compute_sigma(sensitivity)
                signal_magnitude = torch.abs(avg_update).mean().item()
                noise_scale = min(sigma, 0.01 * signal_magnitude) if signal_magnitude > 0 else sigma
                
            else:
                # Light privacy
                sensitivity = self.config.clip_norm / n_clients
                noise_scale = self._compute_sigma(sensitivity)
            
            # Add Gaussian noise
            if noise_scale > 0 and self.config.epsilon_per_round < 1000:
                noise = torch.randn_like(avg_update) * noise_scale
                avg_update = avg_update + noise
            
            aggregated[key] = avg_update
        
        return aggregated

    def update_global_model(self, aggregated_update: Dict):
        """Apply update with trust region and small LR"""
        if not aggregated_update:
            return
        
        # Compute global update norm
        sq = 0.0
        for k, v in aggregated_update.items():
            sq += float(v.detach().pow(2).sum().item())
        global_norm = np.sqrt(sq) + 1e-12
        
        
        eps = self.config.epsilon_per_round
        dp_enabled = eps < 100
        
        # Adjust parameters based on DP mode
        if dp_enabled:
            cap = self.config.server_update_clip  # 3.0
            lr = self.config.server_lr  # 0.005
        else:
            # No-DP: allow larger updates
            cap = self.config.server_update_clip * 2.0  # 6.0
            lr = self.config.server_lr * 1.5  # 0.0075
            print(f"  No-DP mode: Using lr={lr:.4f}, clip={cap:.1f}")
        
        # Trust region clipping
        scale = min(1.0, cap / global_norm)
        
        if scale < 1.0:
            print(f"  Trust region: clipping update from {global_norm:.2f} to {cap:.2f}")
        
        # Apply update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_update:
                    update = aggregated_update[name].to(param.device, param.dtype)
                    # param.add_(-lr * scale * update)
                    param.add_(-lr * scale * update)
    
        ema_decay = getattr(self.config, 'ema_decay', 0.0)
        if ema_decay > 0:
            if not hasattr(self, 'ema_state'):
                self.ema_state = {name: param.data.clone() 
                                for name, param in self.model.named_parameters()}
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    self.ema_state[name] = ema_decay * self.ema_state[name] + (1-ema_decay) * param.data
                    param.data.copy_(self.ema_state[name])
        
        self.round_num += 1
        
        if self.round_num % 10 == 0:
            print(f"  Server: LR={lr:.4f}, scale={scale:.3f}, global_norm={global_norm:.2f}")


    def _check_prediction_diversity(self):
        """Check if model is predicting diverse classes"""
        self.model.eval()
        with torch.no_grad():
            # Generate random test inputs
            if hasattr(self.model, 'fc1'):
                input_dim = self.model.fc1.in_features
            else:
                
                input_dim = 62  #
            
            # Test with various inputs
            test_batch_size = 100
            test_inputs = torch.randn(test_batch_size, input_dim)
            
           
            test_inputs[25:50] = torch.randn(25, input_dim) * 0.1  # Small inputs
            test_inputs[50:75] = torch.randn(25, input_dim) * 10   # Large inputs
            test_inputs[75:100] = torch.zeros(25, input_dim)       # Zero inputs
            
            outputs = self.model(test_inputs)
            predictions = outputs.argmax(dim=1)
            unique_predictions = len(torch.unique(predictions))
            
            pred_counts = torch.bincount(predictions)
            majority_class = predictions[0].item()  n
            
        self.model.train()
        
        return {
            'unique_predictions': unique_predictions,
            'majority_class': majority_class,
            'prediction_distribution': pred_counts.tolist() if len(pred_counts) <= 10 else None
        }

    def save_best_model(self, accuracy: float):
        """Save model if best so far AND diverse"""
        diversity = self._check_prediction_diversity()
        
        # Only save if model is diverse AND accurate
        if accuracy > self.best_accuracy and diversity['unique_predictions'] > 1:
            self.best_accuracy = accuracy
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            return True
        return False

    def load_best_model(self):
        """Load the best model found"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model: accuracy={self.best_accuracy:.4f}")
            
            # Verify diversity after loading
            diversity = self._check_prediction_diversity()
            if diversity['unique_predictions'] == 1:
                print("WARNING: Best model also collapsed, keeping current")
                # Don't actually load collapsed model
                return False
        return True


    def get_noise_multiplier(self):
        """Return sigma_t for Gaussian mechanism"""
        epsilon = self.config.epsilon_per_round
        
        # For large epsilon, return 0 (no noise)
        if epsilon >= 100:
            return 0.0
        
        if epsilon < 0.01:
            epsilon = 0.01
        
        delta = self.config.delta
        gaussian_factor = np.sqrt(2 * np.log(1.25 / delta))
        return gaussian_factor / epsilon

    
    def adaptive_clipping(self, pre_clip_norms: List[float]):
        """Privacy-aware adaptive clipping"""
        if not pre_clip_norms or not self.config.adaptive_clipping:
            return
        
        # Filter outliers
        filtered_norms = [n for n in pre_clip_norms if 0 < n < 100]
        
        if len(filtered_norms) < 3:
            return
        
        if self.config.epsilon_per_round <= 1.0:
           
            target_percentile = 50  # Median
        else:
            target_percentile = 75
        
        percentile_value = np.percentile(filtered_norms, target_percentile)
        
        alpha = 0.1
        old_clip = self.config.clip_norm
        new_clip = alpha * percentile_value * 1.2 + (1 - alpha) * old_clip
        

        if self.config.epsilon_per_round <= 1.0:
            new_clip = np.clip(new_clip, 0.1, 1.0)  
        else:
            new_clip = np.clip(new_clip, 0.5, 5.0)
        
        if abs(new_clip - old_clip) > 0.05:
            self.config.clip_norm = new_clip
            if self.round_num % 10 == 0:
                print(f"  Adaptive clip: {old_clip:.2f} -> {new_clip:.2f}")
    

    def _compute_sigma(self, sensitivity: float) -> float:
        """Compute noise scale for differential privacy"""
        if self.config.epsilon_per_round >= 1000:
            return 0
        
        epsilon = self.config.epsilon_per_round
        delta = self.config.delta
        
        # Standard Gaussian mechanism
        base_sigma = sensitivity * np.sqrt(2 * np.log(1.25/delta)) / epsilon
    
        return base_sigma

    
    def save_best_model(self, accuracy: float):
        """Save model if best so far"""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            return True
        return False
    
    def load_best_model(self):
        """Load the best model found"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model: accuracy={self.best_accuracy:.4f}")