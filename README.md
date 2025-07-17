# Scientific Computing Using Python – High Performance Computing

Project by Marcell Fekete for Aalborg University

### Mandelbrot set

It is a quadratic complex mapping:

$z_{i+1} = z_i^2 + c, \quad i = 0, 1, \ldots, I-1$,

where $c \in C$ is a point in the complex plane and $z_i \in C$ for $i = 0, 1, \ldots, I$. The initial condition is $z_0 = 0 + j \cdot 0$ and $z_1, \ldots, z_I$ are the iteratively achieved outputs.

For each observed complex point $c$, we compute $I$ iterations of the equation above and determine the following:

$\imath(c) = \min\mathbb{T}, \quad \mathbb{T} = \{ i \mid |z_i| > T, i = 1, 2, \ldots,I\} \cup \{I\} $

where T is a threshold value and the initial condition is always $|z_0| \leq T$: $1 \leq \imath(c) \leq I$. So if $|z_{i+1}| \leq T, \forall i = 0, \ldots, I-1$. So $T=\emptyset \cup \{I\} \Rightarrow \imath(c) = I$

For computational purposes, we just need to set $\imath(c)$ to the smallest $i+1$ that leads to $|z_{i+1}| > T$:

$\mathcal{M}(c) = \dfrac{\imath(c)}{I}, \quad 0 < \mathcal{M}(c) \leq 1$

Smaller $\mathcal{M}(c)$ values indicate faster progress of a specific complex point $c$ increases $|z_{i+1}|$. A point $c$ belongs to a Mandelbrot set if $|z_{n+1}|$ remains bounded for $n \rightarrow \infty$.

_Numberphile_:

> When iterations are stable (they don't blow up if you keep squaring them), then those numbers ($c$) can be part of the Mandelbrot set.
> You just iterate again and again, and find out what happens in the long term. $\mathcal{M}(c)$ gives information on the stability.

#### Task

Determine $\mathcal{M}(c)$ for a $c$-mesh which we limit: $-2 \leq \mathfrak{R}\{c\} \leq 1$ and $-1.5 \leq \mathfrak{I}\{c\} \leq 1.5$.
We need to then select a number of points for each of $\mathfrak{R}\{c\}$ and $\mathfrak{I}\{c\}$ as $p_{\mathrm{re}}$ and $p_{\mathrm{im}}$.

$\mathbf{C} = \left[ \begin{array}{rrr} -2.0 & \ldots & 1.0 \\ \vdots & & \vdots \\ -2.0 & \ldots & 1.0 \end{array} \right] + j \cdot \left[ \begin{array}{rrr} 1.5 & \ldots & 1.5 \\ \vdots & & \vdots \\ -1.5 & \ldots & -1.5 \end{array} \right] \in \mathbb{C^{p_{re} \times p_{im}}}$

$p_{re}$ and $p_{im}$ should be selected according to the computational resources available and the desired resolution. Good valeus can be $p_{re} = 5000$ and $p_{im} = 5000$ with a threshold of $T = 2$.

***

### Checklist:

1. Implementation

A minimum of three versions:
* A naive version
* A vectorised numba, cython or f2py version
* A multiprocessing version where the user can select a number of processing units

Also validate for correctness.

2. Output

Plots for the Mandelbrot sets, e.g., a colourmap using `matplotlib.pyplot.cm.hot`. Also compare execution time and in the case of the parallel version, execution time and speed-up versus number of units. Figures should be saved in PDF and relevant simulation data should be also saved including the output $\mathbf{z}$.

3. Software design

Explain considerations for overall design of modules and functions, as well as an algorithm. Also considerations for data types, parameter passing, etc.

4. Test plan

Brief explanation of test plan, to show I've tried implementing testing.

5. Profiling and benchmarking

`time.time`
