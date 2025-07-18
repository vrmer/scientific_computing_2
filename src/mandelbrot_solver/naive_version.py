import numpy as np
from .parameters import Parameters


def mandelbrot_eq(z: int|float, c: complex) -> complex:
    r"""
    Applies the Mandelbrot quadratic mapping: $z_{i+1} = z_i^2 + c$.
    
    Parameters:
        z (int | float): The current value of $z$ in the iteration.
        c (complex): The constant complex parameter to add.
        
    Returns:
        complex: The result of applying the Mandelbrot mapping to z.
    """
    return z**2 + c


def iteration(c: complex, I: int, T: int|float) -> float:
    r"""
    Performs Mandelbrot iteration for a single complex point \( c \), using 
    the recurrence relation \( z_{i+1} = z_i^2 + c \) with \( z_0 = 0 \).
    
    The iteration stops when the magnitude of \( z \) exceeds the threshold \( T \),
    or when the maximum number of iterations \( I \) is reached.
    
    Parameters:
        c (complex): The complex point to test for membership in the Mandelbrot set.
        I (int): Maximum number of iterations to perform.
        T (int | float): Escape threshold (typically 2.0).
        
    Returns:
        float: A normalized escape value in the range [0.0, 1.0], where 1.0 means
               the point did not escape within the iteration limit.
    """
    z = 0
    for i in range(1, I + 1):
        z = mandelbrot_eq(z, c)
        if abs(z) > T:
            return i / I
    return 1.0


def compute_mandelbrot_set_naive(C: np.ndarray, params: Parameters):
    """
    Computes the Mandelbrot set for an array of complex values using a naive iteration approach.
    
    Each element in the input array is tested using the Mandelbrot recurrence,
    and a normalized escape value is calculated.
    
    Parameters:
        C (np.ndarray): A 2D NumPy array of complex values representing the grid of complex numbers.
        params: An object with attributes:
            - I (int): Maximum number of iterations.
            - T (int | float): Escape threshold.
        params (Parameters): Dataclass handling passing arguments to various functions.
    
    Returns:
        np.ndarray: A 2D array of the same shape as `C`, containing normalized escape values.
    """
    output_array = np.zeros(C.shape[0] * C.shape[1])
    for idx, c in enumerate(C.ravel()):
        output_array[idx] = iteration(c, params.I, params.T)
    return output_array.reshape(C.shape[0], C.shape[1])
