# Scientific Computing Using Python – High Performance Computing

Project by Marcell Fekete for Aalborg University

#### Mandelbrot set

It is a quadratic complex mapping:

$z_{i+1} = z_i^2 + c, \quad i = 0, 1, \ldots, I-1$,

where $c \in C$ is a point in the complex plane and $z_i \in C$ for $i = 0, 1, \ldots, I$. The initial condition is $z_0 = 0 + j \cdot 0$ and $z_1, \ldots, z_I$ are the iteratively achieved outputs.

For each observed complex point $c$, we compute $I$ iterations of the equation above and determine the following:

$\imath(c) = \min\mathbb{T}, \quad \mathbb{T} = \{ i \mid |z_i| > T, i = 1, 2, \ldots,I\} \cup \{I\} $

where T is a threshold value and the initial condition is always $|z_0| \leq T$: $1 \leq \imath(c) \leq I$. So if $|z_{i+1}| \leq T, \forall i = 0, \ldots, I-1$. So $T=\emptyset \cup \{I\} \Rightarrow \imath(c) = I$

For computational purposes, we just need to set $\imath(c)$ to the smallest $i+1$ that leads to $|z_{i+1}| > T$:

$\mathcal{M}(c) = \dfrac{\imath(c)}{I}, \quad 0 < \mathcal{M}(c) \leq 1$

Smaller $\mathcal{M}(c)$ values indicate faster progress of a specific complex point $c$ increases $|z_{i+1}|$. A point $c$ belongs to a Mandelbrot set if $|z_{n+1}|$ remains bounded for $n \rightarrow \infty$.

See more details about the project by reading the project report: `report.pdf`

## Installation

Clone the GitHub directory using the following command:

```
git clone https://github.com/vrmer/scientific_computing_2.git
```

Create a new `conda` environment from the attached `environment.yaml` file and initialise it:

```
conda env create -f environment.yaml
conda activate scientific_computing
```

Install the `mandelbrot_solver` package from the root directory of the repository.

```
pip install -e .
```

## Usage

To generate and plot the Mandelbrot set while saving all $\mathcal{M}(c)$ values for a target grid $C$ (also generated via the script), run the following script with a config file from the `configs` directory:

```
python main.py configs/naive.toml
```

This provides the following parameters:

* $p_\mathrm{re} = 5000$
* $p_\mathrm{im} = 5000$
* $I = 100$
* $T = 2$
* `real_val_lims` = $[-2.0, 1.0]$
* `imag_val_lims` = $[1.5, -1.5]$
* `figure_dir` = "figures/"
* `output_dir` = "output/"
* `backend` = "naive"

## Testing

Running the `pytest` command carries out 8 tests in the `src/tests` directory, validating the naive, vectorised, and distributed implementations and comparing their behaviours. The tests also generate 5 warnings, connected to using the client functionality in the `Dask` library.

## Directory Structure

#### `configs` directory

It contains the configuration files in the toml format that are used to input parameters to the Mandelbrot implementations. The backend parameter allows passing "naive", "numba", or "dask" for the naive, the vectorised, and the distributed solution, respectively.

#### `figures` directory

It contains the plots generated from the Mandelbrot solutions.

#### `logs` directory

Log files containing the timing of the various implementations. In the case of `Dask`, the filenames also contain the number of workers (`w`) and threads per worker (`t`) used.

#### `src` directory

It contains the source files of the repository in its subfolder, the `mandelbrot_solver`, as well as test files.

##### `mandelbrot_solver` directory

* `__init__.py`
* `dask_version.py`: function computing the distributed solution
* `meshgrid.py`: function initialising the meshgrid $C$
* `naive_version.py`: functions computing the naive solution
* `numba_version.py`: functions computing the vectorised solution
* `parameters.py`: definition of the `Parameters` dataclass that handles variables
* `utils.py`: helper function to allow for parameters in the config file to fill the parameters of the dataclass in `parameters.py`
* `visualise.py`: function for plotting the Mandelbrot sets

##### `tests` directory

* `test_points.py`: test the behaviour of the solutions with respect to known points
* `test_properties.py`: test the properties of the resulting Mandelbrot sets, e.g., symmetry with respect to the real axis

#### `.gitignore`

Files and directories to be ignored by the GitHub repository.

#### `dask_notebook.ipynb`

Notebook in which I was trying to solve the `dask` implementation.

#### `dask_profile_output.txt`

Results of the `dask` profiling.

#### `environment.yaml`

YAML file to initialise the conda environment from, for installation, see the section [Installation](#installation).

#### `main.py`

The main script of the repository (see [Usage](#usage)).

#### `main.py.lprof`

Non-human readable profiling file.

#### `mandelbrot.ipynb`

Notebook in which I was trying to solve the various implementations.

#### `naive_profile_output.txt` and `numba_profile_output.txt`

Results of the `naive` and `numba` profiling.

#### `report.pdf`

Full report on the project.

#### `setup.py`

It install the `mandelbrot_solver` package (see [Installation](#installation)).
