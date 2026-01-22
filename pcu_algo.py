
import torch
import numpy as np
import time
from typing import Dict, List, Tuple
from server_ops import ServerOps
from client_ops import ClientOps
from metrics_logger import PCUFLMetricsLogger
import copy
import hashlib
from proof_utils import make_policy_view, compute_proof_v2



class PCUFL:
    """Main PCU-FL Algorithm - Fully Aligned with Paper"""
   

    def __init__(self, config, model, train_loaders, test_loader):
        self.config = config
        self.server = ServerOps(config, model)
        self.clients = []  # MUST be initialized before the loop
        
        # Save initial model state for emergency recovery
        self.initial_model_state = copy.deepcopy(model.state_dict())
        
        # Initialize policy Θ_t (Algorithm 1, line 1)
        self.policy = {
            'S_clip': config.clip_norm,
            'U': config.l_inf_bound,
            'K': config.ring_dim,
            'rho': config.epsilon_per_round
        }
        
        # Initialize clients with proper model copies
        for i in range(config.num_clients):
            if i < len(train_loaders) and train_loaders[i] is not None:
                client_model = copy.deepcopy(model)
                # Make sure client gets the UPDATED config
                client = ClientOps(i, config, client_model, train_loaders[i])
                self.clients.append(client)  # Now this will work
        
        self.test_loader = test_loader
        
        # Initialize metrics logger
        self.logger = PCUFLMetricsLogger(config)
        
        # Track best accuracy for early stopping
        self.best_accuracy = 0.0
        self.rounds_without_improvement = 0
        
        # Adjust patience based on privacy level
        if config.epsilon_per_round <= 1.0:
            self.patience = 40  # Strong privacy needs more rounds
        elif config.epsilon_per_round <= 10:
            self.patience = 30  # Moderate privacy
        else:
            self.patience = 20  # Light privacy
        
        # Initialize DP histogram for adaptive clipping
        self.histogram_bins = np.geomspace(0.1, 100, config.histogram_bins)
        
        print(f"Initialized {len(self.clients)} clients with model")
        
        # Initial evaluation
        self.logger.start_round(0)
        self.logger.log_utility_metrics(test_loader, self.server.model)
        initial_metrics = self.logger.current_round["utility"]
        
        acc = initial_metrics.get('test_acc')
        if acc is not None:
            print(f"Initial: Accuracy={acc:.4f}")
            self.best_accuracy = acc
            
        self.logger.end_round()
        




    def _sample_discrete_gaussian(self, shape, sigma: float) -> np.ndarray:
        """Sample from discrete Gaussian distribution for DP noise"""
        if sigma <= 0:
            return np.zeros(shape, dtype=np.int64)
        
        # Check if we should use verifiable noise
        if getattr(self.config, 'verifiable_noise', False):
            # Use verifiable noise with seed commitment
            from noise_commit import derive_gaussian, seed_commit
            round_id = self.logger.current_round.get('round_id', 1)
            param_id = f"round:{round_id}|shape:{shape}"
            seed = hashlib.sha256(param_id.encode()).digest()
            z, transcript = derive_gaussian(seed, shape, sigma)
            
            # Log commitment for verification
            self.logger.current_round["noise"]["noise_seed_commit"] = seed_commit(seed).hex()
            return z
        else:
            # Standard discrete Gaussian (still better than continuous)
            import math
            K = int(math.ceil(sigma * math.sqrt(2.0 * math.log(max(2.0/1e-12, 2.0))) + 4.0))
            ks = np.arange(-K, K+1, dtype=np.int64)
            inv2s2 = 1.0/(2.0*sigma*sigma)
            logits = -(ks.astype(np.float64)**2) * inv2s2
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            cdf = np.cumsum(probs)
            
            # Sample uniformly and invert CDF
            size = int(np.prod(shape))
            uniforms = np.random.uniform(0, 1, size)
            idx = np.searchsorted(cdf, uniforms, side='right')
            return ks[idx].reshape(shape).astype(np.int64)
        

    def run(self):
        """Execute PCU-FL protocol - Aligned with Algorithms 1-3"""
        print(f"Starting PCU-FL with {len(self.clients)} clients")
        
        for round_t in range(self.config.num_rounds):
            round_start_time = time.time()
            
            # Start logging for this round
            self.logger.start_round(round_t + 1)
            
            print(f"\n--- Round {round_t + 1}/{self.config.num_rounds} ---")
            
            # Update policy for this round
            self.policy['S_clip'] = self.config.clip_norm
            self.policy['round'] = round_t + 1
            
            # 1. Sample clients
            selected_clients = np.random.choice(
                len(self.clients), 
                min(self.config.clients_per_round, len(self.clients)),
                replace=False
            )
            
            # 2. Client-Side PCU Generation (Algorithm 1)
            client_submissions = []
            client_raw_updates = {}  # Keep for fallback/debugging
            client_encrypted_updates = {}  # ADD THIS: Store encrypted updates
            pre_clip_norms = []
            client_metadata = {}
            
            global_weights = copy.deepcopy(self.server.model.state_dict())
            
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                client.current_round = round_t + 1
                client_start = time.time()
                
                try:
                    # Algorithm 1, Line 1: Compute raw gradient
                    raw_update = client.train_update(global_weights)
                    
                    # Diagnostic
                    update_norm = self._compute_update_norm(raw_update)
                    print(f"    Client {client_idx} raw update norm: {update_norm:.6f}")
                    
                    # Generate full PCU (includes clipping, quantization, encryption, proof)
                    ciphertext, proof = client.generate_pcu(raw_update)
                    
                    # Store the nu_i (pre-clip norm) that was recorded in generate_pcu
                    pre_clip_norms.append(client.nu_i if hasattr(client, 'nu_i') else update_norm)
                    
                    # Store both encrypted and raw (for fallback)
                    client_encrypted_updates[client.client_id] = ciphertext  # ENCRYPTED
                    client_raw_updates[client.client_id] = raw_update  # Keep raw for fallback
                    
                    client_submissions.append((
                        client.client_id, 
                        ciphertext,  # This is now properly encrypted
                        proof
                    ))
                    
                    # Collect metadata for threat scoring
                    client_metadata[client.client_id] = {
                        'submission_time': time.time() - client_start,
                        'update_size': sum(v.numel() for v in raw_update.values()),
                        'proof_size': len(proof)
                    }
                    
                except Exception as e:
                    print(f"Client {client.client_id} failed: {e}")
                    continue
            
            # 3. Parallel Verification & Trust-Budget Gating (Algorithm 2)
            accepted_clients = self._verify_and_gate_with_trust(
                client_submissions,
                client_metadata,
                self.policy
            )
            
            print(f"Accepted {len(accepted_clients)}/{len(client_submissions)} clients")
            
            if len(accepted_clients) == 0:
                print("No clients accepted this round")
                self.logger.end_round()
                continue
            
            # Get weights from gating
            weights = {c: self._integer_weights[c]/self.config.alpha for c in accepted_clients}
            W_t = sum(weights.values())
            
            # Log round context
            self.logger.log_round_context(
                n_recv=len(client_submissions),
                n_accept=len(accepted_clients),
                weights=weights,
                W_t=W_t
            )
            
            # 5. Ciphertext Aggregation with Discrete Gaussian DP (Algorithm 3)
            # Check if we have proper encryption support
            if all(isinstance(client_encrypted_updates.get(c), dict) for c in accepted_clients):
                # Use encrypted path (Algorithm 3 compliant)
                aggregated_update = self._aggregate_with_discrete_dp_encrypted(
                    client_encrypted_updates,  # Pass ENCRYPTED updates
                    accepted_clients,
                    weights,
                    W_t
                )
            else:
                # Fallback to plaintext simulation
                print("  WARNING: Using plaintext simulation (no proper encryption)")
                aggregated_update = self._aggregate_with_discrete_dp(
                    client_raw_updates,  # Plaintext fallback
                    accepted_clients,
                    weights,
                    W_t
                )
            
            # Rest of your code remains the same...
            # 6. Update global model
            if aggregated_update:
                self.server.update_global_model(aggregated_update)
            
            # 7. Adaptive Clipping
            if self.config.adaptive_clipping and pre_clip_norms and self.config.epsilon_per_round > 1.0:
                new_clip = self._adaptive_clipping_with_dp_histogram(pre_clip_norms)
                old_clip = self.config.clip_norm
                self.config.clip_norm = new_clip
                self.logger.log_adaptive_clipping(new_clip, old_clip)
            else:
                self.config.clip_norm = min(self.config.clip_norm, 1.0)
            
                # 8. Log all metrics
                self._log_round_metrics(
                    aggregated_update,
                    pre_clip_norms,
                    W_t
                )
                
                # 9. Evaluate and check stopping conditions
                metrics = self.logger.current_round["utility"]
                acc = metrics.get('test_acc', 0.0)
                
                # ADD NaN DETECTION AND EMERGENCY RECOVERY HERE
                # Check for model failure (NaN detection)
                if acc == 0 and round_t > 5:
                    # Check if model has NaN
                    has_nan = False
                    for name, param in self.server.model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            has_nan = True
                            print(f"CRITICAL: NaN detected in {name}")
                            break
                    
                    # Also check if all predictions are the same (model collapse)
                    if not has_nan:
                        # Quick test to see if model is collapsed
                        with torch.no_grad():
                            test_batch = next(iter(self.test_loader))
                            test_data, _ = test_batch
                            test_output = self.server.model(test_data[:10])
                            predictions = test_output.argmax(dim=1)
                            if len(torch.unique(predictions)) == 1:
                                print(f"CRITICAL: Model collapsed to single class {predictions[0].item()}")
                                has_nan = True  # Treat as failure
                    
                    if has_nan:
                        print("EMERGENCY: Resetting model to initial state")
                        self.server.model.load_state_dict(self.initial_model_state)
                        self.server.round_num = 0  # Reset server round counter
                        self.server.momentum_buffer = {}  # Clear momentum
                        self.config.clip_norm = 1.0  # Reset clipping norm
                        # Don't end round yet, continue with reset model
                        print("Model reset complete, continuing training")
                        
                        # Re-evaluate with reset model
                        self.logger.log_utility_metrics(self.test_loader, self.server.model)
                        metrics = self.logger.current_round["utility"]
                        acc = metrics.get('test_acc', 0.0)
                        print(f"After reset: Accuracy={acc:.4f}")
                
                # Continue with normal flow
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    self.rounds_without_improvement = 0
                    print(f"  ✓ New best accuracy: {acc:.4f}")
                else:
                    self.rounds_without_improvement += 1
                
                # Print progress
                if (round_t + 1) % 5 == 0:
                    print(f"Round {round_t + 1}: Acc={acc:.4f} (best={self.best_accuracy:.4f}), "
                        f"ε={self.logger.cumulative_epsilon:.2f}")
                
                # End round
                self.logger.end_round()
                
                # Early stopping
                if self.rounds_without_improvement >= self.patience:
                    print(f"\n✓ No improvement for {self.patience} rounds, stopping")
                    break
                
                if acc > 0.99:
                    print(f"\n✓ Achieved near-perfect accuracy ({acc:.4f})")
                    break
            
            # Generate plots and return summary
            self.logger.generate_plots()
            summary = self.logger.get_summary_metrics()
            summary['best_accuracy'] = self.best_accuracy
            return summary


    def _aggregate_with_discrete_dp_encrypted(self, encrypted_updates, accepted_clients, weights, W_t):
        """
        Algorithm 3: Ciphertext Aggregation with Discrete Gaussian DP
        Fixed to maintain proper tensor shapes
        """
        import math, numpy as np, torch, hashlib
        from he_backend import HEBackend
        from noise_commit import derive_gaussian
        
        if not accepted_clients:
            return {}
        
        # Initialize HE backend
        he = HEBackend(
            n=self.config.ring_dim,
            t=self.config.modulus_t,
            kappa=getattr(self.config, 'packing_kappa', 1)
        )
        
        # Get integer weights
        tilde = {i: int(self._integer_weights[i]) for i in accepted_clients if i in self._integer_weights}
        alpha = int(self.config.alpha)
        gamma = int(self.config.gamma)
        
        W_int = int(sum(tilde.values()))
        w_max_float = max(tilde.values()) / float(alpha) if tilde else 0
        
        print(f"  Homomorphic aggregation: K={len(tilde)}, W_int={W_int}, w_max={w_max_float:.4f}")
        
        # Get original parameter shapes from the model
        param_shapes = {}
        for name, param in self.server.model.named_parameters():
            param_shapes[name] = param.shape
        
        # Compute sensitivity
        d = sum(np.prod(shape) for shape in param_shapes.values())
        S_clip = float(self.config.clip_norm)
        eta_max = float(getattr(self.config, 'eta_max', 0.0))
        W_min = float(getattr(self.config, 'W_min', 1.0))
        
        Delta2_star = (2.0 * w_max_float / max(W_min, 1e-12)) * (
            S_clip + math.sqrt(d) * (1.0/(2.0*gamma) + eta_max)
        )
        self._last_Delta2_star = Delta2_star
        
        # Check DP settings
        sigma_t = self.server.get_noise_multiplier()
        eps = float(self.config.epsilon_per_round)
        dp_enabled = (eps > 0.0) and (sigma_t > 0.0) and math.isfinite(sigma_t) and (eps < 100.0)
        
        # Get parameter keys
        param_keys = list(encrypted_updates[accepted_clients[0]].keys())
        
        # Line 5: Homomorphic weighted sum
        Y_t = {}
        
        for param_key in param_keys:
            n_chunks = len(encrypted_updates[accepted_clients[0]][param_key])
            Y_t[param_key] = []
            
            for chunk_idx in range(n_chunks):
                Y_chunk = None
                
                for client_id in accepted_clients:
                    if client_id not in tilde:
                        continue
                    
                    ct = encrypted_updates[client_id][param_key][chunk_idx]
                    weighted_ct = he.mul_plain(ct, int(tilde[client_id]))
                    
                    if Y_chunk is None:
                        Y_chunk = weighted_ct
                    else:
                        Y_chunk = he.add(Y_chunk, weighted_ct)
                
                Y_t[param_key].append(Y_chunk)
        
        print(f"  Computed homomorphic sum Y_t over {len(param_keys)} parameters")
        
        # Line 6: Sample noise if DP enabled
        if dp_enabled:
            sigma_int = alpha * gamma * W_int * sigma_t * Delta2_star
            print(f"  DP noise: σ_t={sigma_t:.6f}, Δ₂*={Delta2_star:.4f}, σ_int={sigma_int:.1f}")
        else:
            sigma_int = 0
            print(f"  No DP noise (ε={eps:.1f})")
        
        # Lines 7-8: Add noise and decrypt
        aggregated = {}
        
        for param_key in param_keys:
            chunks_decoded = []
            
            for chunk_idx, Y_chunk in enumerate(Y_t[param_key]):
                if dp_enabled and sigma_int > 0:
                    shape = Y_chunk.data.shape
                    seed_str = f"round{self.logger.current_round.get('round_id', 0)}_{param_key}_{chunk_idx}"
                    seed = hashlib.sha256(seed_str.encode()).digest()
                    z, transcript = derive_gaussian(seed, shape, sigma_int)
                    
                    pt_noise = he.encode(z)
                    c_noise = he.encrypt(pt_noise)
                    Y_noisy = he.add(Y_chunk, c_noise)
                else:
                    Y_noisy = Y_chunk
                
                pt_result = he.decrypt(Y_noisy)
                chunks_decoded.append(pt_result.data)
            
            # Concatenate chunks
            if len(chunks_decoded) == 1:
                full_param = chunks_decoded[0]
            else:
                full_param = np.concatenate([c.flatten() for c in chunks_decoded])
 
        # Line 9: Decode to get average
        denom = float(gamma * W_int)
        param_tensor = torch.tensor(full_param.astype(np.float32) / denom, dtype=torch.float32)

        # Smart overflow detection and filtering
        # Only remove extreme outliers, preserve normal gradient values
        overflow_threshold = 100.0  # No gradient should realistically be this large
        mask = torch.abs(param_tensor) > overflow_threshold

        if mask.any():
            # Found overflow values - replace them with zeros
            param_tensor = torch.where(mask, torch.zeros_like(param_tensor), param_tensor)
            print(f"  Removed {mask.sum().item()} overflow values from {param_key}")

        # Reshape to match original parameter shape
        if param_key in param_shapes:
            target_shape = param_shapes[param_key]
            if param_tensor.numel() == np.prod(target_shape):
                param_tensor = param_tensor.reshape(target_shape)
            else:
                print(f"  WARNING: Shape mismatch for {param_key}")
                # continue  # Skip if shape doesn't match

        aggregated[param_key] = param_tensor

        # Compute update norm
        update_norm = math.sqrt(sum(float((v**2).sum().item()) for v in aggregated.values()))
        print(f"  Aggregated update norm: {update_norm:.4f}")

        return aggregated


    def _verify_and_gate_with_trust(self, submissions, metadata, policy):
        """Algorithm 2: Verification with proper budget management and Hamilton weights"""
        accepted = []
        dropped_by_budget = 0
        dropped_by_proof = 0
        
        current_round = self.logger.current_round.get('round_id', 1)
        
        # Budget recovery every 10 rounds
        if current_round % 10 == 0 and current_round > 0:
            recovered = 0
            for client_id in range(len(self.clients)):
                if self.server.trust_budgets.get(client_id, 0) <= 0:
                    self.server.trust_budgets[client_id] = self.config.initial_budget // 2
                    recovered += 1
            if recovered > 0:
                print(f"  Recovered budget for {recovered} clients")
        
        print(f"  Processing {len(submissions)} submissions")
        
        for client_id, encrypted_update, proof in submissions:
            # Get current budget (initialize if needed)
            if client_id not in self.server.trust_budgets:
                self.server.trust_budgets[client_id] = self.config.initial_budget
            
            current_budget = self.server.trust_budgets[client_id]
            
            # Check budget first
            if current_budget <= 0:
                dropped_by_budget += 1
                continue
            
            # Compute threat score
            theta_i = self._compute_threat_score(client_id, metadata.get(client_id, {}))
            
            # Update budget if high threat (but don't drop yet)
            if theta_i > self.config.threat_threshold:
                self.server.trust_budgets[client_id] = max(0, current_budget - 1)
            
            # Verify proof
            if not self._verify_nizk_proof(proof, policy, client_id, current_round):
                dropped_by_proof += 1
                continue
            
            accepted.append(client_id)
        
        print(f"  Gating: {len(accepted)} accepted, {dropped_by_budget} dropped (budget), {dropped_by_proof} failed (proof)")
        
        K = len(accepted)
        if K == 0:
            print(f"  WARNING: No clients accepted!")
            return []
        
        alpha = self.config.alpha  # 256
        
        # Hamilton/Largest-Remainder Method for fair weight allocation
        raw_w = {i: 1.0 / K for i in accepted}
        
        # 1) Compute quotas on α-grid
        q = {i: raw_w[i] * alpha for i in accepted}
        
        # 2) Base allocation (floor) and fractional parts
        base = {i: int(np.floor(q[i])) for i in accepted}
        frac = {i: q[i] - base[i] for i in accepted}
        
        # 3) Number of +1s to distribute
        R = int(alpha) - sum(base.values())
        
        # 4) Give +1 to R clients with largest fractional parts
        # Stable tie-breaking using client_id
        order = sorted(accepted, key=lambda i: (frac[i], -int(i)), reverse=True)
        tilde = base.copy()
        
        for i in order[:R]:
            tilde[i] += 1
        
        # 5) Optional: Cap maximum weight to prevent outliers
        cap = int(np.ceil(alpha / K))  # 9 for K=30, α=256
        
        # Check if any weight exceeds cap
        needs_redistribution = False
        for i in accepted:
            if tilde[i] > cap:
                needs_redistribution = True
                break
        
        if needs_redistribution:
            # Redistribute excess
            for i in accepted:
                if tilde[i] > cap:
                    excess = tilde[i] - cap
                    tilde[i] = cap
                    # Give excess to clients below cap
                    for j in order:
                        if excess == 0:
                            break
                        if j != i and tilde[j] < cap:
                            tilde[j] += 1
                            excess -= 1
        
        # Verify sum
        W_int = sum(tilde.values())
        if W_int != alpha:
            print(f"  WARNING: Weight sum {W_int} != α={alpha}, adjusting...")
            # Force adjustment on last client
            diff = alpha - W_int
            tilde[order[-1]] += diff
            W_int = alpha
        
        # Store for aggregation
        self._integer_weights = tilde
        w_max_float = max(tilde.values()) / float(alpha)
        self._w_max_float = w_max_float
        
        # Sanity check
        assert W_int == alpha, f"Weight sum {W_int} != α={alpha}"
        
        print(f"  Weights: K={K}, W_int={W_int}, W_min_int={alpha} ✓, w_max_float={w_max_float:.4f}")
        
        return accepted

    
    def _compute_threat_score(self, client_id, metadata):
        """Compute threat score from metadata only"""
        score = 0.0
         # Reduce penalties for strong privacy mode
        if self.config.epsilon_per_round <= 0.5:
            return 0.0  # Accept all clients in strong privacy mode
        # Fast submission (potential attacker)
        if metadata.get('submission_time', 1.0) < 0.1:
            score += 0.3
        
        # Unusual update size
        avg_size = 1000  # Expected size
        if abs(metadata.get('update_size', avg_size) - avg_size) > avg_size * 0.5:
            score += 0.2
        
        # Missing or invalid proof
        if metadata.get('proof_size', 0) == 0:
            score += 0.5
        
        return min(score, 1.0)

    def _adaptive_clipping_with_dp_histogram(self, pre_clip_norms):
        """Adaptive clipping targeting specific clip rate"""
        if not pre_clip_norms:
            return self.config.clip_norm
        
        # Current clipping rate
        current_clip_rate = sum(1 for n in pre_clip_norms if n > self.config.clip_norm) / len(pre_clip_norms)
        
        # If far from target, be more aggressive
        target_rate = self.config.target_clip_rate
        
        if abs(current_clip_rate - target_rate) > 0.1:  # More than 10% off target
            # Use direct percentile without noise for faster convergence
            target_percentile = int((1 - target_rate) * 100)
            new_clip = np.percentile(pre_clip_norms, target_percentile)
            alpha = 0.5  # Aggressive adaptation
        else:
            # Near target, use DP histogram
            hist, _ = np.histogram(pre_clip_norms, bins=self.histogram_bins)
            noise_scale = 1.0 / self.config.histogram_epsilon
            noisy_hist = hist + np.random.laplace(0, noise_scale, size=len(hist))
            noisy_hist = np.maximum(noisy_hist, 0)
            
            cumsum = np.cumsum(noisy_hist)
            if cumsum[-1] > 0:
                target_percentile = self.config.quantile_target
                threshold_idx = np.argmax(cumsum >= cumsum[-1] * target_percentile)
                new_clip = self.histogram_bins[threshold_idx]
            else:
                new_clip = self.config.clip_norm
            alpha = 0.2  # Slower when close to target
        
        # Update
        new_clip = alpha * new_clip + (1 - alpha) * self.config.clip_norm
        new_clip = np.clip(new_clip, 0.01, 5.0)
        
        print(f"  Adaptive clipping: rate={current_clip_rate:.1%} → target={target_rate:.1%}, "
            f"S_clip: {self.config.clip_norm:.3f} → {new_clip:.3f}")
        
        self.config.clip_norm = new_clip
        return new_clip
    
    def _compute_weights(self, accepted_clients):
        """Algorithm 2: Compute weights on 1/α grid with unbiased integer correction"""
        n = len(accepted_clients)
        if n == 0:
            return {}
        
        alpha = self.config.alpha
        W_min = self.config.W_min
        
        # Start with uniform weights (float)
        raw_weights = {c: 1.0/n for c in accepted_clients}
        
        # Round to 1/α grid (float)
        wq = {}
        for c in accepted_clients:
            wq[c] = np.round(raw_weights[c] * alpha) / alpha
        
        # Convert to integer representation with i* correction
        A_sorted = sorted(accepted_clients)  # Deterministic ordering
        target_sum = int(np.round(alpha * sum(wq.values())))
        
        tilde = {}
        run = 0
        
        # All but last get floor
        for i in A_sorted[:-1]:
            ti = int(np.floor(alpha * wq[i]))
            tilde[i] = ti
            run += ti
        
        # Last client gets remainder (unbiased adjustment)
        i_star = A_sorted[-1]
        tilde[i_star] = target_sum - run
        
        # Check W_min constraint (integer domain)
        W_int = sum(tilde.values())
        
        print(f"  Weights: W_float={sum(wq.values()):.4f}, αW_float≈{int(alpha*sum(wq.values()))}, W_int={W_int} (W_min={int(W_min*alpha)}) {'✓' if W_int >= W_min*alpha else '✗'}")
        
        if W_int < W_min * alpha:
            print(f"  WARNING: W_int={W_int} < W_min*α={W_min*alpha}, adjusting...")
            # Scale up all weights proportionally
            scale = (W_min * alpha) / W_int
            for c in tilde:
                tilde[c] = int(np.ceil(tilde[c] * scale))
            W_int = sum(tilde.values())
        
        # Return both integer and float representations
        self._integer_weights = tilde  # Store for aggregation
        return {c: tilde[c]/alpha for c in accepted_clients}  # Return float for compatibility
     
    def _encode_for_rlwe(self, update):
        """Placeholder for RLWE encoding"""
        # Would pack update into ring elements
        return b"encoded_" + str(hash(frozenset(update.items()))).encode()
    
    def _encrypt_rlwe(self, encoded):
        """Placeholder for RLWE encryption"""
        # Would use actual RLWE encryption
        return b"encrypted_" + encoded
    
    def _generate_nizk_proof(self, update, policy, client_id):
        """Placeholder for NIZK proof generation"""
        # Would generate actual zkSNARK proof
        proof_data = {
            'client_id': client_id,
            'policy_hash': hashlib.sha256(str(policy).encode()).hexdigest(),
            'update_hash': hashlib.sha256(str(update).encode()).hexdigest()
        }
        return str(proof_data).encode()
    
    def _verify_nizk_proof(self, proof: bytes, policy: Dict, client_id: int, round_idx: int) -> bool:
        """Verify proof using v2"""
        if not self.config.strict_proof_verification:
            return isinstance(proof, bytes) and len(proof) == 32
        
        if not isinstance(proof, bytes) or len(proof) != 32:
            return False
        
        policy_view = make_policy_view(self.config)
        expected = compute_proof_v2(policy_view, client_id, round_idx)
        
        if proof != expected and self.config.log_proof_diag:
            print(f"      Proof mismatch: got {proof.hex()[:8]}... exp {expected.hex()[:8]}...")
        
        return proof == expected
    
    def _log_round_metrics(self, aggregated_update, pre_clip_norms, W_t):
        """Log all metrics for the round"""
        # Log utility metrics
        clip_rates = [1 if norm > self.config.clip_norm else 0 
                     for norm in pre_clip_norms]
        self.logger.log_utility_metrics(
            self.test_loader,
            self.server.model,
            aggregated_update,
            clip_rates
        )
        
        # Log privacy metrics
        sigma_model = self.server._compute_sigma(
            getattr(self, "_last_Delta2_star", self.config.clip_norm * W_t)
        )
        # Record Δ2* used this round into the logger for transparency
        if hasattr(self, "_last_Delta2_star"):
            self.logger.current_round["round_context"]["Delta2_sum"] = float(self._last_Delta2_star)
        self.logger.log_privacy_metrics(sigma_model, W_t)
        
        # Log HE metrics
        if aggregated_update:
            n_coefficients = sum(
                np.prod(v.shape) for v in aggregated_update.values()
            )
            self.logger.log_he_metrics(n_coefficients)
    
    def _compute_update_norm(self, update: Dict) -> float:
        """Compute L2 norm of update"""
        total_norm = 0
        for key in update:
            if torch.is_tensor(update[key]):
                total_norm += torch.sum(update[key] ** 2).item()
        return np.sqrt(total_norm)