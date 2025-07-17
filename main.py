import os
import time
import h5py
import logging
import tomllib
import argparse
from src.mandelbrot_solver import (
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
    
    parser.add_argument("-w", "--n_workers",
                        help="Number of workers, if not given, value from config file is used.",
                        required=False, type=int)
    parser.add_argument("-t", "--threads_per_worker",
                        help="Number of threads per worker, if not given, value from config file is used.",
                        required=False, type=int)
    parser.add_argument("--do_profiling", action="store_true", help="If used, don't save outputs")
    
    args = parser.parse_args()
    
    # load parameters
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
        
    params = filter_to_dataclass(Parameters, config)
    
    # optionally override n_workers and threads_per_worker
    params.n_workers = args.n_workers if args.n_workers else params.n_workers
    params.threads_per_worker = args.threads_per_worker if args.threads_per_worker else params.threads_per_worker
    
    log_dir = "logs/"
    
    # set up profiling if revelant
    if args.do_profiling:
        log_dir = "logs/profiling"
        
    params.output_dir = None
    params.figure_dir = None
    
    # set up logging
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=(f"{log_dir}{params.backend}_w{params.n_workers}_t{params.threads_per_worker}.log" 
                  if params.backend == "dask" else f"{log_dir}/{params.backend}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # initialise the meshgrid
    meshgrid = init_meshgrid(params)
    # select the target function and calculate the mandelbrot set
    start_time = time.time()
    mandelbrot_set = mandelbrot_set_compute_dict[params.backend](meshgrid, params)
    end_time = time.time()
    # retain the mandelbrot set
    if params.output_dir:
        os.makedirs(params.output_dir, exist_ok=True)
        outpath = os.path.join(params.output_dir, f"{params.backend}.hdf5")
        with h5py.File(outpath, "w") as f:
            f.create_dataset("mandelbrot_escapes", data=mandelbrot_set)
    # plot the mandelbrot set
    plot_mandelbrot_set(mandelbrot_set, params)
    
    speed = round(end_time-start_time, 4)
    
    logging.info(f"Mandelbrot set computed using {params.backend} in {speed} secs.")
    if params.backend == "dask":
        logging.info(f"Performance achieved via {params.n_workers} workers and {params.threads_per_worker} threads per worker.")
        logging.info(f"Speed up compared to using the naive implementation: {round(72.682 / speed, 4)} times")
