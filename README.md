# Polarized Consensus-Based Optimization

![PolarGIF](https://user-images.githubusercontent.com/44805883/201196111-d4dcc1c3-4ee9-47df-927a-e03659c990cd.gif)

Polarized swarm dynamics for optimization and sampling.

## What is PolarCBO?

Polarized consensus-based dynamics allow to apply consensus-based optimization (CBO) and sampling (CBS) for objective functions with several global minima or distributions with many modes, respectively. Here we have 

* particles $\{x^{(i)}\}\in\mathbb{R}^d$ which explore the space,
* the objective $V:\mathbb{R}^d\to\mathbb{R}$ which we want to optimize.

The position of the particles are updated via the stochastic ODE

$$
\begin{align}
    \boxed{%
    d x^{(i)} = -(x^{(i)} - m(x^{(i)})) d t + \sigma |x^{(i)} - m(x^{(i)})| d W^{(i)}
    }
\end{align}
$$

where

* $W^{(i)}$ are independent Brownian motions
* $\sigma$ scales the influence of the noise term

The choice of $m$ is the heart of PolarCBO

