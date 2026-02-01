import ctypes
import os
import time
import numpy as np


def load_library():
    # assumes libmatrix.so is in the same directory as this script.
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(here, "libmatrix.so")

    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Could not find {lib_path}. Make sure you compiled Step 7.2:\n"
            "  nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so"
        )

    return ctypes.cdll.LoadLibrary(lib_path)


def main():
    lib = load_library()

# contacts ctypes
    lib.gpu_matrix_multiply.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]
    lib.gpu_matrix_multiply.restype = None 

    N = 1024

    # create input matrices and output buffer
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    A_flat = np.ascontiguousarray(A.ravel())
    B_flat = np.ascontiguousarray(B.ravel())
    C_flat = np.ascontiguousarray(C.ravel())

    lib.gpu_matrix_multiply(A_flat, B_flat, C_flat, N)

    # timed run
    start = time.perf_counter()
    lib.gpu_matrix_multiply(A_flat, B_flat, C_flat, N)
    end = time.perf_counter()

    C_result = C_flat.reshape(N, N)

    print(f"Python call to CUDA library completed in {end - start:.6f} seconds (N={N})")

if __name__ == "__main__":
    main()
