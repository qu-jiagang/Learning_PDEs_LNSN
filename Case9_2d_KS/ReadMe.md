### Case 8: Two-dimensional Burgers equation with reactive source

$$
& u_t + \Delta u + \Delta^2 u + \frac{1}{2}|\nabla u|^2 = 0 \\
            & u(x,y,t_0) = \sin(x+y)+\sin(x) +\sin(y)
$$

#### Numerical details

* $(x,y,t)\in[0,L_x]\times[0,L_x]\times[0,T]$
* $\nu_1=(\pi/L_x)^2$，$\nu_2=(\pi/L_y)^2$
* $\nu_1=\nu_2=0.1$
* T = 1000 s, $\delta t=0.0025$ s
* Mesh: uniform mesh of $64\times 64$

#### Initial condition

* $u(x,y,t_0) = \sin(x+y)+\sin(x) +\sin(y)$

#### Files introduction

* `2d_KS.py`:  Numerical solutions for 2d KS equation, periodic boundary, Fourier Spectral method.
* `LNSN.py`:  Linear convolutional neural network
  * `validation_LNSN.py`:  validation for LNSN
* `CNN.py`:  Convolutional neural network
  * `validation_CNN.py`:  validation for CNN
* `PLOT.py`: plot contours (one of test data samples)

#### Figure



