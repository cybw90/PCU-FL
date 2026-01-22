# proof_utils.py
import hashlib
import json
import struct

def _float_str(x: float) -> str:
    """Fixed formatting for floats to ensure consistency"""
    return format(float(x), ".17g")

def make_policy_view(cfg_or_dict) -> dict:
    """Extract exact fields for proof - MUST be identical on both sides"""
    # Support either config object or dict
    def get(k, default=None):
        if hasattr(cfg_or_dict, k):
            return getattr(cfg_or_dict, k)
        elif isinstance(cfg_or_dict, dict):
            return cfg_or_dict.get(k, default)
        return default
    
    view = {
        "version": "pcufl-proof-v2",
        "clip_norm": _float_str(get("clip_norm", 1.0)),
        "alpha": int(get("alpha", 256)),
        "gamma": int(get("gamma", 256)),
        "ring_dim": int(get("ring_dim", 4096)),
        "packing_kappa": int(get("packing_kappa", 1)),
        "fp8_mode": str(get("fp8_mode", "none")),
        "epsilon_per_round": _float_str(get("epsilon_per_round", 10.0)),
        "delta": _float_str(get("delta", 1e-5)),
    }
    return view

def canonical_policy_bytes(policy_view: dict) -> bytes:
    """Stable JSON serialization"""
    return json.dumps(policy_view, sort_keys=True, separators=(",", ":")).encode("utf-8")

def compute_proof_v2(policy_view: dict, client_id, round_idx: int) -> bytes:
    """Compute deterministic proof"""
    tag = b"PCUFL|proof-v2|"
    pol = canonical_policy_bytes(policy_view)
    cid = str(client_id).encode("utf-8")
    rid = struct.pack(">Q", int(round_idx))
    return hashlib.sha256(tag + pol + cid + rid).digest()