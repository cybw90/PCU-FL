
import numpy as np, hmac, hashlib, math

def seed_commit(seed: bytes) -> bytes:
    return hashlib.sha256(seed).digest()

def _drbg_stream(seed: bytes, n_bytes: int) -> bytes:
    out = bytearray(); counter = 0
    while len(out) < n_bytes:
        msg = counter.to_bytes(8, 'big')
        out.extend(hmac.new(seed, msg, hashlib.sha256).digest()); counter += 1
    return bytes(out[:n_bytes])

def _uniform01_from_bytes(bb: bytes) -> np.ndarray:
    arr = np.frombuffer(bb, dtype=np.uint64)
    return (arr / np.float64(2**64 - 1)).astype(np.float64)

def _sample_discrete_gaussian(shape, sigma: float, uniforms: np.ndarray) -> np.ndarray:
    if sigma <= 0: return np.zeros(shape, dtype=np.int64)
    size = int(np.prod(shape))
    uniforms = uniforms[:size]
    # Truncate at K std devs (tight tail bound)
    K = int(math.ceil(sigma * math.sqrt(2.0 * math.log(max(2.0/1e-12, 2.0))) + 4.0))
    ks = np.arange(-K, K+1, dtype=np.int64)
    probs = np.exp(-(ks.astype(np.float64)**2) / (2.0 * sigma * sigma))
    probs /= probs.sum()
    cdf = np.cumsum(probs)
    idx = np.searchsorted(cdf, uniforms, side='right')
    return ks[idx].reshape(shape).astype(np.int64)

def derive_gaussian(seed: bytes, shape, sigma_int: float):
    size = int(np.prod(shape))
    bb = _drbg_stream(seed, 8*size)
    u = _uniform01_from_bytes(bb)
    z = _sample_discrete_gaussian(shape, float(sigma_int), u)
    transcript = {
        "commit": seed_commit(seed).hex(),
        "shape": tuple(int(s) for s in shape),
        "sigma_int": float(sigma_int),
        "counter_bytes": 8,
        "engine": "HMAC-SHA256-DRBG-invcdf",
    }
    return z, transcript

def verify_transcript(commit_hex: str, transcript: dict, z: np.ndarray) -> bool:
    try:
        if transcript.get("commit") != commit_hex: return False
        if tuple(z.shape) != tuple(transcript["shape"]): return False
        if "sigma_int" not in transcript: return False
        return True
    except Exception:
        return False
