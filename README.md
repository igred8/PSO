# Particle Swarm Optimizer (PSO)

This is a basic implementation of a PSO in python36. 

Tha algorithm is well described by this [overview article by Poli, Kennedy, and Blackwell](https://doi.org/10.1007/s11721-007-0002-0). This implementation follows "algorithm 1" in the paper with "construction coefficients".

The algorithm is a high-level iteration procedure that tries to improve a set of solutions so that they minimize a cost function. PSO does not require specific knowledge of the function being optimized. This allows the algorithm to be applied to a wide array of problems that can be defined by a function, $f(\vec{x}) = \vec{y}$. This is particularly useful in problems where the function is not differentiable or differentiation is expensive. However, PSO does not guarantee an optimum will be found and may perform worse for problems where methods like gradient descent may be used. 

![flowchart](https://github.com/igred8/PSO/blob/master/pso_flowchart.png)

## 2020.05.28
The current example finds the optimum parameters of a quadrupole triplet that produce a focused electron beam at a specified location.
