
import json
from time import time
import torch
import numpy as np
from typing import Tuple, Dict
import copy
import hashlib
from proof_utils import make_policy_view, compute_proof_v2
from he_backend import HEBackend

class ClientOps:
    """Client-side operations for PCU-FL - Algorithm 1 compliant"""
    
    def __init__(self, client_id, config, model, data_loader):
        self.client_id = client_id
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.trust_budget = config.initial_budget

            # Add HE backend
        
        self.he = HEBackend(
            n=config.ring_dim,
            t=config.modulus_t,
            kappa=getattr(config, 'packing_kappa', 1)
        )
        self.current_round = 0



    def train_update(self, global_weights: Dict) -> Dict:
        """Fix: Use reasonable learning rate"""
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        initial_weights = {
            key: val.clone().detach() 
            for key, val in self.model.state_dict().items()
        }
        
        lr = 0.2
        max_batches = 20
        
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr,
            momentum=0.5,
            weight_decay=1e-4
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            if batch_idx >= max_batches:
                break
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
    # Compute update
        update = {}
        zero_params = []
        tiny_params = []  
        
        with torch.no_grad():
            for key in self.model.state_dict():
                update[key] = self.model.state_dict()[key] - initial_weights[key]
                
                # Check for zero updates
                if torch.sum(torch.abs(update[key])) == 0:
                    zero_params.append(key)
                # Check for updates that will round to zero with gamma=256
                elif torch.abs(update[key]).max().item() < 1.0/256:
                    tiny_params.append((key, torch.abs(update[key]).max().item()))
        
        # Report findings for first client only
        if self.client_id == 0:
            if zero_params:
                print(f"    Client 0: Zero updates for: {zero_params}")
            if tiny_params:
                for name, max_val in tiny_params:
                    print(f"    Client 0: {name} max update={max_val:.6f} < 1/256, will become zero!")
        
        return update


    def clip_update(self, update: dict) -> tuple:
        """Algorithm 1 Line 2: Use exact S_clip from config"""
        # Compute pre-clip norm
        total_norm_sq = 0
        for key in update:
            if torch.is_tensor(update[key]):
                total_norm_sq += torch.sum(update[key] ** 2).item()
        
        nu_i = np.sqrt(total_norm_sq)
        
        # Use EXACT clip threshold from policy (don't cap at 1.0)
        S_clip = self.config.clip_norm
        
        if nu_i > S_clip:
            scale = S_clip / nu_i
            clipped_update = {k: v * scale for k, v in update.items()}
        else:
            clipped_update = {k: v.clone() for k, v in update.items()}
        
        # L∞ bound enforcement
        U = self.config.l_inf_bound
        for key in clipped_update:
            clipped_update[key] = torch.clamp(clipped_update[key], -U, U)
        
        return clipped_update, nu_i


    def quantize_fp8(self, update: Dict) -> Dict:
        """Algorithm 1 Step 3-4: FP8 quantization with validity checks"""
        if self.config.fp8_mode == "none":
            return update
        
        quantized = {}
        max_attempts = 3
        scale_factor = 1.0
        
        for attempt in range(max_attempts):
            all_valid = True
            temp_quantized = {}
            
            for key, tensor in update.items():
                # Scale tensor for better range usage
                scaled_tensor = tensor * scale_factor
                
                # Check pre-quantization validity
                if torch.isnan(scaled_tensor).any() or torch.isinf(scaled_tensor).any():
                    raise ValueError(f"Invalid values in {key} before quantization")
                
                # Apply quantization
                if self.config.fp8_mode == "E4M3":
                    q_tensor = self._quantize_e4m3_proper(scaled_tensor)
                elif self.config.fp8_mode == "E5M2":
                    q_tensor = self._quantize_e5m2_proper(scaled_tensor)
                else:
                    q_tensor = scaled_tensor
                
                # Check for saturation (Line 4 of Algorithm 1)
                if self._is_saturated(q_tensor, self.config.fp8_mode):
                    all_valid = False
                    scale_factor *= 0.5  # Reduce scale for next attempt
                    break
                
                temp_quantized[key] = q_tensor / scale_factor  # Unscale
            
            if all_valid:
                quantized = temp_quantized
                break
        
        if not all_valid:
            raise ValueError(f"Cannot quantize without saturation after {max_attempts} attempts")
        
        return quantized
    
    def _quantize_e4m3_proper(self, tensor):
        """Simple FP8 E4M3 quantization"""
        # Basic quantization - just reduce precision
        scale = tensor.abs().max() / 240.0  # E4M3 max value
        if scale > 0:
            quantized = (tensor / scale).round().clamp(-240, 240)
            return quantized * scale
        return tensor

    def _is_saturated(self, tensor: torch.Tensor, fp8_mode: str) -> bool:
        """Check if FP8 values are saturated"""
        if fp8_mode == "E4M3":
            max_val = 448.0
            threshold = max_val * 0.95  # 95% of max is considered saturated
        elif fp8_mode == "E5M2":
            max_val = 57344.0
            threshold = max_val * 0.95
        else:
            return False
        
        return torch.max(torch.abs(tensor)) > threshold

    # In client_ops.py
    def encode_and_pack(self, quantized_update: dict) -> dict:
        """Algorithm 1 Line 5: Handle multi-ring packing while preserving shapes"""
        packed = {}
        gamma = self.config.gamma
        K = self.config.ring_dim
        kappa = self.config.packing_kappa
        slot_capacity = K * kappa
        modulus_t = self.config.modulus_t
        
        for key, tensor in quantized_update.items():
            # Store original shape for reconstruction
            original_shape = tensor.shape
            
            # Fixed-point encoding
            fixed_point = torch.round(tensor * gamma).to(torch.int64)
            
            # Check bounds
            if torch.max(torch.abs(fixed_point)) > modulus_t // 2:
                raise ValueError(f"Values exceed ring modulus in {key}")
            
            # Flatten for packing
            flat = fixed_point.flatten()
            
            # Pack into chunks
            chunks = []
            for i in range(0, flat.numel(), slot_capacity):
                chunk = flat[i:i+slot_capacity]
                chunks.append(chunk.clone())
            
            # Store chunks with shape info
            packed[key] = {
                'chunks': chunks,
                'original_shape': original_shape
            }
        
        return packed

    def _encrypt_rlwe(self, packed_update: dict) -> dict:
        """Encrypt while preserving shape metadata"""
        encrypted = {}
        
        for key, pack_info in packed_update.items():
            encrypted[key] = []
            
            # If packed_update has the new format with shape info
            if isinstance(pack_info, dict) and 'chunks' in pack_info:
                chunks = pack_info['chunks']
            else:
                # Fallback for old format
                chunks = pack_info if isinstance(pack_info, list) else [pack_info]
            
            for chunk in chunks:
                chunk_np = chunk.cpu().numpy()
                pt = self.he.encode(chunk_np)
                ct = self.he.encrypt(pt)
                encrypted[key].append(ct)
        
        return encrypted

    def generate_pcu(self, update: Dict) -> Tuple[bytes, bytes]:
        """Generate PCU following Algorithm 1"""
        # Line 2: Clip and record norm
        clipped_update, nu_i = self.clip_update(update)
        self.nu_i = nu_i  # STORE for the main loop to access
        
        # Line 3: FP8 quantization (if enabled)
        if self.config.fp8_mode != "none":
            quantized_update = self.quantize_fp8(clipped_update)
        else:
            quantized_update = clipped_update
        
        # Line 5: Encode and pack
        packed_update = self.encode_and_pack(quantized_update)
        
        # Line 6: Encrypt
        ciphertext = self._encrypt_rlwe(packed_update)
        
        # Line 7: Generate proof
        from proof_utils import make_policy_view, compute_proof_v2
        policy_view = make_policy_view(self.config)
        proof = compute_proof_v2(policy_view, self.client_id, self.current_round)
        
        return ciphertext, proof


    
    def _generate_proof(self, packed_update: Dict, nu_i: float) -> bytes:
        """Generate proof that matches server's expectation"""
        import hashlib, json, struct
        
        # Helper to format floats consistently
        def _float_str(x: float) -> str:
            return format(float(x), ".17g")
        
        # Create policy view matching server's expectation
        policy_view = {
            "version": "pcufl-proof-v2",
            "clip_norm": _float_str(self.config.clip_norm),
            "alpha": int(self.config.alpha),
            "gamma": int(self.config.gamma),
            "W_min_int": int(self.config.alpha),  # Server uses alpha as W_min_int
            "ring_dim": int(self.config.ring_dim),
            "packing_kappa": int(getattr(self.config, 'packing_kappa', 1)),
            "fp8_mode": str(self.config.fp8_mode),
            "epsilon_per_round": _float_str(self.config.epsilon_per_round),
            "delta": _float_str(self.config.delta),
        }
        
        # Compute proof matching server's compute_proof_v2
        tag = b"PCUFL|proof-v2|"
        body = json.dumps(policy_view, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cid = str(self.client_id).encode("utf-8")
        round_idx = getattr(self, 'current_round', 0)
        rid = struct.pack(">Q", int(round_idx))
        
        proof = hashlib.sha256(tag + body + cid + rid).digest()
        return proof


    def _quantize_e4m3(self, tensor: torch.Tensor) -> torch.Tensor:
        """E4M3 quantization with improved range usage"""
        max_val = 448.0
        # FIX: Use 50% of range instead of 10%
        target_max = max_val * 0.5
        
        abs_max = torch.max(torch.abs(tensor)).item()
        if abs_max > 0:
            scale = min(target_max / abs_max, 100.0)
        else:
            return tensor
        
        scaled = tensor * scale
        quantum = (2 * target_max) / 240  # E4M3 has ~240 levels
        quantized = torch.round(scaled / quantum) * quantum
        quantized = torch.clamp(quantized, -target_max, target_max)
        
        return quantized / scale

    def _quantize_e5m2(self, tensor: torch.Tensor) -> torch.Tensor:
        """E5M2 quantization with improved range usage"""
        max_val = 57344.0
      
        target_max = max_val * 0.05
        
        abs_max = torch.max(torch.abs(tensor)).item()
        if abs_max > 0:
            scale = min(target_max / abs_max, 100.0)
        else:
            return tensor
        
        scaled = tensor * scale
        quantum = (2 * target_max) / 120  # E5M2 has ~120 levels
        quantized = torch.round(scaled / quantum) * quantum
        quantized = torch.clamp(quantized, -target_max, target_max)
        
        return quantized / scale

    def _check_saturation(self, tensor: torch.Tensor, fp8_mode: str) -> bool:
        """Check if tensor is saturated"""
        if fp8_mode == "E4M3":
            threshold = 448.0 * 0.5 * 0.95  # 95% of usable range
        elif fp8_mode == "E5M2":
            threshold = 57344.0 * 0.05 * 0.95
        else:
            return False
        
        max_abs = torch.max(torch.abs(tensor)).item()
        return max_abs >= threshold

    def _compute_norm(self, update: Dict) -> float:
        """Compute L2 norm of update"""
        total_norm = 0
        for key in update:
            if torch.is_tensor(update[key]):
                tensor = update[key].float() if update[key].dtype != torch.float32 else update[key]
                total_norm += torch.sum(tensor ** 2).item()
        return np.sqrt(total_norm)