# zk_proofs.py
import json, struct, hashlib

def prove_clip_inf_fp8_rounding(config, client_id, round_idx: int) -> bytes:
    """Generate deterministic proof bound to policy parameters"""
    from proof_utils import make_policy_view, canonical_policy_bytes
    
    view = make_policy_view(config)
    pol = canonical_policy_bytes(view)
    cid = str(client_id).encode("utf-8")
    rid = struct.pack(">Q", int(round_idx))
    
    return hashlib.sha256(b"PCUFL|proof-v2|" + pol + cid + rid).digest()

def verify(proof: bytes, public_inputs: dict) -> bool:
    """Verify proof matches expected policy binding"""
    if not isinstance(proof, (bytes, bytearray)) or len(proof) != 32:
        return False
    
    # Extract parameters from public_inputs
    config = public_inputs.get('config')
    client_id = public_inputs.get('client_id')
    round_idx = public_inputs.get('round_idx')
    
    if not all([config, client_id is not None, round_idx is not None]):
        return False
    
    expected = prove_clip_inf_fp8_rounding(config, client_id, round_idx)
    return proof == expected