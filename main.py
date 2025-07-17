import time
import tomllib
import argparse
from src import (
    init_meshgrid, compute_mandelbrot_set_naive, 
    compute_mandelbrot_set_vectorized_wrapper, plot_mandelbrot_set, 
    Parameters, compute_mandelbrot_set_distributed, filter_to_dataclass)
    


mandelbrot_set_compute_dict = {
    "naive": compute_mandelbrot_set_naive,
    "numba": compute_mandelbrot_set_vectorized_wrapper,
    "dask": compute_mandelbrot_set_distributed
}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""Config with parameters for computing the Mandelbrot set""")
    
    parser.add_argument("config", help="Input toml file containing all necessary parameters.")
    
    args = parser.parse_args()
    
    # load parameters
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
        
    params = filter_to_dataclass(Parameters, config)

    # initialise the meshgrid
    meshgrid = init_meshgrid(params)
    # select the target function and calculate the mandelbrot set
    start_time = time.time()
    mandelbrot_set = mandelbrot_set_compute_dict[params.backend](meshgrid, params)
    end_time = time.time()
    # plot the mandelbrot set
    plot_mandelbrot_set(mandelbrot_set, params)
    
    print(f"\nMandelbrot set computed using  {params.backend}  in  {round(end_time-start_time, 4)}  secs.")
