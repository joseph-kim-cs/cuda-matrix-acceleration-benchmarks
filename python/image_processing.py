import os
import time
import ctypes
import numpy as np

# generating images through numpy, just 32x32 images of various patterns
def make_gradient(N: int) -> np.ndarray:
    # horizontal gradient
    x = np.linspace(0, 1, N, dtype=np.float32)
    img = np.tile(x, (N, 1))
    return img

def make_checkerboard(N: int, block: int = 16) -> np.ndarray:
    # checkerboard
    y = np.arange(N) // block
    x = np.arange(N) // block
    grid = (y[:, None] + x[None, :]) % 2
    return grid.astype(np.float32)

def make_gaussian_blob(N: int, sigma: float = 0.12) -> np.ndarray:
    # 2d guassian blob
    yy, xx = np.meshgrid(np.linspace(-1, 1, N, dtype=np.float32),
                         np.linspace(-1, 1, N, dtype=np.float32),
                         indexing="ij")
    rr2 = xx * xx + yy * yy
    return np.exp(-rr2 / (2 * sigma * sigma)).astype(np.float32)

def add_noise(img: np.ndarray, noise_std: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img + rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)

# functions to create filters
def box_blur(K: int) -> np.ndarray:
    assert K % 2 == 1
    ker = np.ones((K, K), dtype=np.float32)
    ker /= ker.sum()
    return ker

def gaussian_blur(K: int, sigma: float) -> np.ndarray:
    assert K % 2 == 1
    r = K // 2
    y, x = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    ker = np.exp(-(x*x + y*y) / (2 * sigma * sigma)).astype(np.float32)
    ker /= ker.sum()
    return ker

def sharpen_3x3() -> np.ndarray:
    return np.array([[0, -1,  0],
                     [-1, 5, -1],
                     [0, -1,  0]], dtype=np.float32)

def edge_laplacian_3x3() -> np.ndarray:
    return np.array([[0,  1, 0],
                     [1, -4, 1],
                     [0,  1, 0]], dtype=np.float32)

def sobel_x() -> np.ndarray:
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

def sobel_y() -> np.ndarray:
    return np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)

# cpu references
def conv2d_same_cpu(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    N = img.shape[0]
    K = ker.shape[0]
    r = K // 2
    out = np.zeros_like(img, dtype=np.float32)

    # naive reference (very slow but correct)
    for y in range(N):
        for x in range(N):
            s = 0.0
            for ky in range(K):
                iy = y + (ky - r)
                if iy < 0 or iy >= N:
                    continue
                for kx in range(K):
                    ix = x + (kx - r)
                    if ix < 0 or ix >= N:
                        continue
                    s += float(img[iy, ix]) * float(ker[ky, kx])
            out[y, x] = s
    return out

# GPU-accelerated convolution via ctypes, loading the CUDA shared library

def load_cuda_lib() -> ctypes.CDLL:
    # expects running from lab3/python
    lib_path = os.path.abspath(os.path.join("..", "cuda", "libmatrix.so"))
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Cannot find libmatrix.so at: {lib_path}")

    lib = ctypes.cdll.LoadLibrary(lib_path)

    lib.gpu_convolution_same.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gpu_convolution_same.restype = None
    return lib

def conv2d_same_gpu(lib: ctypes.CDLL, img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    N = img.shape[0]
    K = ker.shape[0]
    out = np.zeros((N, N), dtype=np.float32)

    lib.gpu_convolution_same(img.ravel(), ker.ravel(), out.ravel(), N, K)
    return out




# Example usage and timing
def main():
    N = 256
    img = add_noise(make_gaussian_blob(N), noise_std=0.05, seed=0)

    # select kernel
    ker = gaussian_blur(K=7, sigma=1.5)  # try box_blur(7), sharpen_3x3(), sobel_x(), etc.

    print(f"Image: {N}x{N}, Kernel: {ker.shape[0]}x{ker.shape[1]}")

    # CPU reference 
    if N <= 256:
        t0 = time.time()
        out_cpu = conv2d_same_cpu(img, ker)
        t1 = time.time()
        print(f"CPU conv time: {(t1 - t0):.4f} s")
    else:
        out_cpu = None
        print("Skipping CPU reference at N>256 (too slow). Use N=128 or 256 to validate first.")

    # GPU
    lib = load_cuda_lib()
    t0 = time.time()
    out_gpu = conv2d_same_gpu(lib, img, ker)
    t1 = time.time()
    print(f"GPU conv (ctypes end-to-end) time: {(t1 - t0):.4f} s")

    if out_cpu is not None:
        max_abs = np.max(np.abs(out_cpu - out_gpu))
        mean_abs = np.mean(np.abs(out_cpu - out_gpu))
        print(f"Max abs diff CPU vs GPU: {max_abs:.6e}")
        print(f"Mean abs diff CPU vs GPU: {mean_abs:.6e}")

    print(f"Output stats: min={out_gpu.min():.4f}, max={out_gpu.max():.4f}, mean={out_gpu.mean():.4f}")

if __name__ == "__main__":
    main()
