import tomllib
import argparse
from dask.distributed import Client
from src import (
    init_meshgrid, compute_mandelbrot_set_naive, 
    compute_mandelbrot_set_vectorized, plot_mandelbrot_set, 
    Parameters, compute_mandelbrot_set_distributed)

# TODO: restructure main to incorporate all solutions from one code
# TODO: run the full naive code
    


mandelbrot_set_compute_dict = {
    "naive": compute_mandelbrot_set_naive,
    "numba": compute_mandelbrot_set_vectorized,
    "dask": compute_mandelbrot_set_distributed
}


if __name__ == "__main__":

    params = Parameters(
        p_re=1000,
        p_im=1000,
        I=100,
        backend="dask"
    )

    # meshgrid = init_meshgrid(params.p_re, params.p_im, params.real_val_lims, params.imag_val_lims)

    meshgrid = init_meshgrid(params)

    # TODO: change to pass the whole parameters
    mandelbrot_set = mandelbrot_set_compute_dict[params.backend](meshgrid, params)
    
    print(mandelbrot_set)

    plot_mandelbrot_set(mandelbrot_set, f"figures/{params.backend}.pdf")
