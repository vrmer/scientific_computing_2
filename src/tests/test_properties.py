from mandelbrot_solver import (Parameters, compute_mandelbrot_set_naive, 
                               compute_mandelbrot_set_vectorized_wrapper, compute_mandelbrot_set_distributed)


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


class TestSymmetry:
    """
    # TODO: docstring
    """
    # ChatGPT inspiration
    def text_real_axis_symmetry_naive(self):
        test_points = [complex(0.3, 0.5), complex(-0.8, 0.2), complex(-1, 0.01)]
        for c in test_points:
            c_conj = c.conjugate()
            z1 = compute_mandelbrot_set_naive(c, params)
            z2 = compute_mandelbrot_set_naive(c_conj, params)
            assert z1 == z2, f"Symmetry failed for {c} and {c_conj}: {z1} != {z2}"
            
    def text_real_axis_symmetry_naive(self):
        test_points = [complex(0.3, 0.5), complex(-0.8, 0.2), complex(-1, 0.01)]
        for c in test_points:
            c_conj = c.conjugate()
            z1 = compute_mandelbrot_set_vectorized_wrapper(c, params)
            z2 = compute_mandelbrot_set_vectorized_wrapper(c_conj, params)
            assert z1 == z2, f"Symmetry failed for {c} and {c_conj}: {z1} != {z2}"
            
    def text_real_axis_symmetry_naive(self):
        test_points = [complex(0.3, 0.5), complex(-0.8, 0.2), complex(-1, 0.01)]
        for c in test_points:
            c_conj = c.conjugate()
            z1 = compute_mandelbrot_set_distributed(c, params)
            z2 = compute_mandelbrot_set_distributed(c_conj, params)
            assert z1 == z2, f"Symmetry failed for {c} and {c_conj}: {z1} != {z2}"
