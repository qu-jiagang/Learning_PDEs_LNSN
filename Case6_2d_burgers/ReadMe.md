### Case 6: Two-dimensional Burgers equation

$$
\boldsymbol{u}_t + \boldsymbol{u}\cdot \nabla\boldsymbol{u} = \nu \Delta\boldsymbol{u} \\
\boldsymbol{u}(x,y,t_0) = \boldsymbol{u}_0(x,y)
$$

#### Numerical details

* $\mu=0.015$

* $\delta t=0.01$
* T = 3 s, steps = 300

* Mesh: uniform mesh of $128\times 128$
* Domain: $[0,2\pi]\times[0,2\pi]$

#### Initial condition

* Similar with Case 5: randomly fields

#### Files introduction

* `2d_burgers_Periodic.py`:  Numerical solutions for 2d burgers equation with periodic boundary, Fourier Spectral method.

* `LNSN.py`:  Linear and nonlinear separate network
  * `LNSN.net`: trained LNSN

* `LNSN.py`:  Linear and nonlinear separate network
  * `LNSN.net`: trained CNN

* `validation_LNSN.py`:  validation for LNSN
  * `error-lnsn.txt`: error for LNSN varying with time
* `validation_CNN.py`:  validation for CNN
  * `error-cnn.txt`: error for CNN varying with time

* `PLOT.py`: plot contours (one of test data samples)

* `report.py`: plot error varying with time

#### Figure

![2d_burgers_u](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case6_2d_burgers\2d_burgers_u.jpg)

![2d_burgers_v](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case6_2d_burgers\2d_burgers_v.jpg)

Figure: Comparison of a short time predictions from LNSN and the standard CNN. The first row shows the numerical results, second row shows the predictions obtained by LNSN, and the last row shows that of CNN.

![2d_burgers_error](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case6_2d_burgers\2d_burgers_error.jpg)

Figure: Prediction error of the LNSN and the standard CNN. The shadow area indicates the minimum and maximum error among $1000$ test samples.