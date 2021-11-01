### Case 7: Two-dimensional reaction diffusion equation

$$
u_t = \mu \Delta u + f(u) \\
f(u) = u-u^3 \\
u(x,y,t_0) = u_0(x,y)
$$

#### Numerical details

* $\mu=0.01$

* $\delta t=0.01$
* T = 3 s, steps = 300

* Mesh: uniform mesh of $128\times 128$
* Domain: $[0,2\pi]\times[0,2\pi]$

#### Initial condition

* Similar with Case 5: randomly fields

#### Files introduction

* `2d_heating_Periodic.py`:  Numerical solutions for 2d reaction diffusion  equation with periodic boundary, Fourier Spectral method.
* `LNSN.py`:  Linear and nonlinear separate network
  * `LNSN.net`: trained LNSN
* `LNSN.py`:  Linear and nonlinear separate network
  * `LNSN.net`: trained CNN

* `validation_LNSN.py`:  validation for LNSN
  * `error-lnsn.txt`: error for LNSN varying with time0.052
* `validation_CNN.py`:  validation for CNN
  * `error-cnn.txt`: error for CNN varying with time

* `PLOT.py`: plot contours (one of test data samples)

* `report.py`: plot error varying with time

#### Figure

![2d_ra](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case7_2d_reaction_diffusion\2d_ra.jpg)

Figure: Comparison of a short time predictions from LNSN and the standard CNN. The first row shows the numerical results, second row shows the predictions obtained by LNSN, and the last row shows that of CNN.

![2d_ra_error](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case7_2d_reaction_diffusion\2d_ra_error.jpg)

Figure: Prediction error of the LNSN and the standard CNN. The shadow area indicates the minimum and maximum error among $1000$ test samples.

