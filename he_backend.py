
import numpy as np
import os
import hmac
import hashlib

class Ciphertext:
    def __init__(self, data: np.ndarray, mask: np.ndarray, t: int):
        self.data = np.asarray(data, dtype=np.int64) % t
        self.mask = np.asarray(mask, dtype=np.int64) % t
        self.t = int(t)
    
    def clone(self):
        return Ciphertext(self.data.copy(), self.mask.copy(), self.t)

class Plaintext:
    def __init__(self, data: np.ndarray, t: int):
        self.data = np.asarray(data, dtype=np.int64) % t
        self.t = int(t)

class HEBackend:
    """Functional homomorphic encryption (NOT cryptographically secure)"""
    
    def __init__(self, n: int = 4096, t: int = 2**31-1, kappa: int = 1, sk: bytes = None):
        self.n = int(n)
        self.t = int(t)
        self.kappa = int(kappa)
        self.sk = sk or os.urandom(32)
        self.pk = b"public_key_placeholder"
    
    def keygen(self):
        return self.pk, self.sk
    
    def encode(self, ints: np.ndarray) -> Plaintext:
        """Fixed: Handle negative values properly"""
        data = np.asarray(ints, dtype=np.int64)

        data = np.where(data < 0, self.t + data, data)
        return Plaintext(data % self.t, self.t)
    
    def decode(self, pt: Plaintext) -> np.ndarray:
        """Fixed: Handle modulus 2^32-5 correctly"""
        data = np.asarray(pt.data, dtype=np.int64)
        

        threshold = 2**31
        
        # Convert large positive to negative
        mask = data > threshold
        data[mask] = data[mask] - self.t
        
        return data
    
    def _prf_mask(self, shape, seed: bytes) -> np.ndarray:
        """Generate deterministic mask"""
        needed = int(np.prod(shape)) * 8
        stream = bytearray()
        ctr = 0
        
        while len(stream) < needed:
            stream.extend(
                hmac.new(self.sk, seed + ctr.to_bytes(8, 'big'), hashlib.sha256).digest()
            )
            ctr += 1
        
        mask = np.frombuffer(bytes(stream[:needed]), dtype=np.uint64).astype(np.int64)
        mask = mask[:int(np.prod(shape))].reshape(shape) % self.t
        return mask
    
    def encrypt(self, pt: Plaintext) -> Ciphertext:
        seed = os.urandom(16)
        mask = self._prf_mask(pt.data.shape, seed)
        return Ciphertext((pt.data + mask) % self.t, mask, self.t)
    
    def decrypt(self, ct: Ciphertext) -> Plaintext:
        """Fixed: Handle modular arithmetic correctly for signed values"""
        # Decrypt
        raw = (ct.data - ct.mask) % self.t
        

        half_t = self.t // 2
        result = np.where(raw > half_t, raw - self.t, raw)
        

        result = np.where(result < 0, self.t + result, result)
        return Plaintext(result, self.t)
    
    def add(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        assert a.t == b.t
        return Ciphertext(
            (a.data + b.data) % a.t,
            (a.mask + b.mask) % a.t,
            a.t
        )
    
    def mul_plain(self, a: Ciphertext, k: int) -> Ciphertext:
        k = int(k) % a.t
        return Ciphertext(
            (a.data * k) % a.t,
            (a.mask * k) % a.t,
            a.t
        )