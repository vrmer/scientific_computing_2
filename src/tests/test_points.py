import numpy as np
from mandelbrot_solver import (Parameters, compute_mandelbrot_set_naive, 
                               compute_mandelbrot_set_vectorized_wrapper, compute_mandelbrot_set_distributed)


# define parameters to use for testing
params = Parameters(
    figure_dir=None,
    output_dir=None,
    p_re=1,
    p_im=1,
    I=100,
    backend=None,
    n_workers=1,
    threads_per_worker=1
)
    

class TestInsideSet:
    """
    [0, -1] should return 1.0 as they belong to the Mandelbrot set.
    """
    def test_naive(self):
        c1 = np.asarray([[0]])
        z1 = compute_mandelbrot_set_naive(c1, params)
        c2 = np.asarray([[-1]])
        z2 = compute_mandelbrot_set_naive(c2, params)
        assert z1.item() and z2.item() == 1.0
        
    def test_vectorized(self):
        c1 = np.asarray([[0]])
        z1 = compute_mandelbrot_set_vectorized_wrapper(c1, params)
        c2 = np.asarray([[-1]])
        z2 = compute_mandelbrot_set_vectorized_wrapper(c2, params)
        assert z1.item() and z2.item() == 1.0
        
    def test_distributed(self):
        c1 = np.asarray([[0]])
        z1 = compute_mandelbrot_set_distributed(c1, params)
        c2 = np.asarray([[-1]])
        z2 = compute_mandelbrot_set_distributed(c2, params)
        assert z1.item() and z2.item() == 1.0
        

class TestOutsideSet:
    """
    [2, 1+1j] should escape quickly.
    """
    def test_naive(self):
        c1 = np.asarray([[2]])
        z1 = compute_mandelbrot_set_naive(c1, params)
        c2 = np.asarray([[1+1j]])
        z2 = compute_mandelbrot_set_naive(c2, params)
        assert 0 < z1.item() < 1 and 0 < z2.item() < 1
        
    def test_vectorized(self):
        c1 = np.asarray([[2]])
        z1 = compute_mandelbrot_set_vectorized_wrapper(c1, params)
        c2 = np.asarray([[1+1j]])
        z2 = compute_mandelbrot_set_vectorized_wrapper(c2, params)
        assert 0 < z1.item() < 1 and 0 < z2.item() < 1
        
    def test_distributed(self):
        c1 = np.asarray([[2]])
        z1 = compute_mandelbrot_set_distributed(c1, params)
        c2 = np.asarray([[1+1j]])
        z2 = compute_mandelbrot_set_distributed(c2, params)
        assert 0 < z1.item() < 1 and 0 < z2.item() < 1
        
        
class TestBoundaryCases:
    """
    [-0.75, 0.001 + 0.001j] are tricky areas near the edge, 
    let's see if the implementations behave similarly
    """
    def test_outputs_1(self):
        c1 = np.asarray([[-0.75]])
        z1_naive = compute_mandelbrot_set_naive(c1, params)
        z1_vectorised = compute_mandelbrot_set_vectorized_wrapper(c1, params)
        z1_distributed = compute_mandelbrot_set_distributed(c1, params)
        assert z1_naive.item() == z1_vectorised.item() == z1_distributed.item()
        
    def test_outputs_2(self):
        c2 = np.asarray([[0.001 + 0.001j]])
        z2_naive = compute_mandelbrot_set_naive(c2, params)
        z2_vectorised = compute_mandelbrot_set_vectorized_wrapper(c2, params)
        z2_distributed = compute_mandelbrot_set_distributed(c2, params)
        assert z2_naive.item() == z2_vectorised.item() == z2_distributed.item()
