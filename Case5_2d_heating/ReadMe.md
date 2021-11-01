### Case 5: Two-dimensional Heat equation

$$
\begin{aligned}
    &\frac{\partial u}{\partial t} = \mu \left( \frac{\partial ^2 u}{\partial x^2} + \frac{\partial ^2 u}{\partial y^2} \right)   \\
    &u(x,0) = f(x)       \\
\end{aligned}
$$

#### Numerical details

* $\mu=0.1$

* $\delta t=0.001$
* T = 3 s, steps = 3000

* Mesh: uniform mesh of $128\times 128$
* Domain: $[0,2\pi]\times[0,2\pi]$

#### Initial condition

* Starting with random fields with $\mu'=0.05$, running for 1 s, $\delta t'=0.001$.

#### Files introduction

* `2d_heating_Periodic.py`:  Numerical solutions for 2d heat equation with periodic boundary, Fourier Spectral method.

* `LCNN.py`:  Linear convolutional neural network
  * `LCNN_ldt.net`: trained network for `dt=0.01`, L means large
  * `LCNN_sdt.net`: trained network for `dt=0.001`, S means small
* `validation_LCNN.py`:  validation
  * `error-ldt.txt`: error with `dt=0.01`
  * `error-sdt.txt`: error with `dt=0.001`

* `PLOT.py`: plot contours (one of test data samples)

* `report.py`: plot error varying with time

![2d_heat](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case5_2d_heating\2d_heat.jpg)

Figure: Comparison of the predictions from numerical simulations and the linear CNN. The first row shows the numerical results, second row shows the predictions with a small time step $\delta t=0.001$, and the last row shows that with a larger time step $\delta t=0.01$.

![2d_heat_error](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case5_2d_heating\2d_heat_error.png)

Figure: Prediction error of the linear CNN for time step value of $\delta t=0.001$ and $\delta t=0.01$. The shadow area indicates the minimum and maximum error among $1000$ test samples.

#### Table

Table: Learned filters

| Time step | Filters for FDM                                              | Filters for CNN                                              |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0.001     | $\left[ \begin{matrix} 0.0069 & 0.0277 & 0.0069 \\ 0.0277 & 0.8617 & 0.0277 \\ 0.0069 & 0.0277 & 0.0069  \end{matrix} \right]$ | $\left[ \begin{matrix} -0.0430 & 0.1282 & -0.0440 \\ 0.1284 & 0.6588 & 0.1303 \\ -0.0441 & 0.1304& -0.0450\end{matrix} \right]$ |
| 0.01      | $\left[ \begin{matrix} 0.0692& 0.2767& 0.0692 \\ 0.2767& -0.3834& 0.2767\\ 0.0692& 0.2767& 0.0692\end{matrix} \right]$ | $\left[ \begin{matrix} 0.1708& 0.6930& 0.1711\\ 0.0695& 0.0390& 0.0691\\ 0.1710& 0.0692& 0.1710\end{matrix} \right]$ |

