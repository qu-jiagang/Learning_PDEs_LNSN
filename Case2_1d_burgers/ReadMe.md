## One-dimensional Burgers equation

$$
\begin{aligned}
	&u_t + u\cdot \nabla u = \frac{1}{Re} \Delta u,\quad (x,t)\in(-\pi,\pi)\times(0,1) \\
	&u(x,t_0) = u_0(x)
\end{aligned}
$$

#### Boundary Conditions: 

1. Dirichlet (zero boundary is shown as an example):  Chebyshev spectral method is used for numerical calculation.

$$
u(-\pi,t)=0;\ u(\pi,t)=0
$$



2. Periodic: Fourier spectral method is used for numerical calculation. The boundary values for $x=-\pi$ and $x=\pi$ are not equal at all. However, the periodic boundary is also encoded into the  Fourier spectral method. 

$$
u(-\pi)=u(\pi)
$$

3. Neumann: Fourier spectral method is used for numerical calculation. Similar to the periodic boundary, the boundary value for the nearest nodes of boundary nodes are not equal, too. 

$$
\frac{du}{dt}|_{x=-\pi}=0;\ \frac{du}{dt}|_{x=\pi}=0
$$

#### Initial Conditions:

There are 3 different initial conditions are considered for Dirichlet and Periodic boundary:

```python
u_initial[0] = -10*x*(1+x)*(1-x)/(1+np.exp(10*x**2))
u_initial[1] = -np.sin(pi*x)
u_initial[2] = -2*x/(1+np.sqrt(1/np.exp(100/8))*np.exp(100*x**2/4))
```

The dataset from first initial condition (```u_initial[0]```) is used for training, and the other are used for validation.

> There are some difficulties in numerical calculation for Neumann boundary. 
>
> ```
> u_initial[0] = 10*x*(1+x)*(1-x)/(1+np.exp(10*x**2))
> u_initial[1] = 2*x/(1+np.sqrt(1/np.exp(100/8))*np.exp(100*x**2/4))
> u_initial[2] = np.sin(0.5*pi*x)
> ```

#### Files introduction:

* `dataset`: folder, dataset and code for generating data
  * `burgers_Dirichlet.py`
  * `burgers_Neumann.py`
  * `burgers_Periodic.py`
* `Dirichlet`/ `Neumnn`/`Periodic`: folders for different boundaries
  * `CNN.py`: standard convolutional neural network 
    * `CNN.net`: trained CNN network
    * `validation_CNN.py`: generate the reconstructions of CNN 
      * `reconst`: folder, including reconstructions by `validation_CNN.py`
  * `LNSN.py:` Linear and nonlinear separated network 
    * `LNSN.net`: trained LNSN network
    * `validation_LNSN.py`: generate the reconstructions of LNSN 
      * `reconst`: folder, including reconstructions by `validation_LNSN.py`
* `PLOT.py`: plot contours

#### Plot:

![1d_burgers](C:\Users\qujiagang\Documents\我的坚果云\Learning_PDEs\Case2_1d_burgers\1d_burgers.jpg)
