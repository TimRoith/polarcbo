# :snowflake: Polarized Consensus-Based Optimization and Sampling

![PolarGIF](https://user-images.githubusercontent.com/44805883/201196111-d4dcc1c3-4ee9-47df-927a-e03659c990cd.gif)

This code produces the examples for the paper "Polarized consensus-based dynamics for optimization and sampling": https://arxiv.org/abs/2211.05238

```
@online{bungert2022polarized,
    author = {Bungert, Leon and Roith, Tim and Wacker, Philipp},
    title = {Polarized consensus-based dynamics for optimization and sampling},
    year = {2022},
    eprint={2211.05238},
    archivePrefix={arXiv},
    primaryClass={math.OC}
}
```

## 💡 What is PolarCBO/CBS?

Polarized consensus-based dynamics allow to apply consensus-based optimization (CBO) and sampling (CBS) for objective functions with several global minima or distributions with many modes, respectively. Here we have 

* particles $\{x^{(i)}\}\in\mathbb{R}^d$ which explore the space,
* the objective $V:\mathbb{R}^d\to\mathbb{R}$ which we want to optimize.

For optimizing $V$ the position of the particles are updated via the stochastic ODE

$$
\begin{align}
    \boxed{%
    d x^{(i)} = -(x^{(i)} - m(x^{(i)})) d t + \sigma |x^{(i)} - m(x^{(i)})| d W^{(i)}
    }
\end{align}
$$

where

* $m(x^{(i)})$ is a weighted empirical mean associated with the point $x^{(i)}$,
* $W^{(i)}$ are independent Brownian motions,
* $\sigma$ scales the influence of the noise term.

For sampling from $\exp(-V)$ the position of the particles are updated via the stochastic ODE

$$
\begin{align}
    \boxed{%
    d x^{(i)} = -(x^{(i)} - m(x^{(i)})) d t + \sqrt{2\lambda^{-1}C(x^{(i)})} d W^{(i)}
    }
\end{align}
$$

where $C(x^{(i)})$ is a weighted empirical covariance matrix associated with the point $x^{(i)}$.

The choice of the functions $m(\dot)$ and $C(\cdot)$ are at the heart of our polarized methods. Given a similarity measure $\mathsf k(\cdot,\cdot)$ and an inverse temperature parameter $\beta>0$ we define

$$
\begin{align}
    m(x) &:= \frac{\sum_{i}\mathsf k(x,x^{(i)})x^{(i)}\exp(-\beta V(x^{(i)}))}{\sum_{i}\mathsf k(x,x^{(i)})\exp(-\beta V(x^{(i)}))},
    \\
    C(x) &:= \frac{\sum_{i}\mathsf k(x,x^{(i)})(x^{(i)}-m(x))\otimes(x^{(i)}-m(x))\exp(-\beta V(x^{(i)}))}{\sum_{i}\mathsf k(x,x^{(i)})\exp(-\beta V(x^{(i)}))}.
\end{align}
$$

Note that these weighted mean and covariance give more influence to particles which are close to $x$ and have a small value of $V$. If $\mathsf k(\cdot,\cdot)=1$ one recovers standard CBO and CBS.


