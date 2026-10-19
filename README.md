# Mathematics for Machine Learning and Deep Learning

An English-language, notebook-first mathematics curriculum for machine learning and deep learning. The material emphasizes concise examples, common pitfalls, numerical checks, and connections to model training rather than long projects.

## Getting started

**Recommended environment:** Python 3.10+ and Jupyter Notebook or JupyterLab.

Install the libraries used throughout the notebooks:

```bash
python -m pip install -r requirements.txt
```

The examples use **NumPy** for numerical computing, **pandas** for tables and experiment logs, **Matplotlib** for vector diagrams and mathematical plots, and **Seaborn** for statistical visualization. No deep-learning framework is required for the core examples. Some advanced ideas are demonstrated with small NumPy implementations so the underlying mathematics remains visible.

Run notebooks from top to bottom. Random examples use fixed seeds where reproducibility matters; plots and numerical approximations may still differ slightly across library versions.

Open the notebooks in numerical order within each folder. Every notebook is designed to be studied independently after its prerequisites.

## Visualization and data-analysis labs

The notebooks include topic-specific visual labs, not long projects. Depending on the chapter, these demonstrate: vector addition and matrix transformations with `matplotlib`; contour plots for gradients and optimization; eigenvector and PCA visualizations; distribution histograms and correlation heatmaps with `seaborn`; labeled result tables and experiment logs with `pandas`; and numerical computations, sampling, and linear algebra with `numpy`.

The plotted data are small synthetic examples intended to build mathematical intuition. Statistical summaries and plots illustrate methods; they are not claims about real-world populations. In exercises involving floating-point numbers or random sampling, small numerical differences are expected.

### A typical workflow

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Use NumPy for the computation, pandas for labeled values,
# and Matplotlib/Seaborn to inspect the result.
vector = np.array([1.0, 2.0, 3.0])
summary = pd.DataFrame({"index": np.arange(vector.size), "value": vector})
print(summary)
sns.barplot(data=summary, x="index", y="value")
plt.title("A small numerical example")
plt.show()
```

## Curriculum

### 1. Linear Algebra
- [Vectors, Norms, and Similarity](1.%20Linear%20Algebra/01_vectors_and_norms.ipynb) — Represent observations and model parameters as vectors, and learn the operations that appear throughout machine learning.
- [Matrices and Linear Systems](1.%20Linear%20Algebra/02_matrices_and_linear_systems.ipynb) — Learn matrix operations, linear systems, rank, determinant, and why solving is usually better than explicitly inverting a matrix.
- [Vector Spaces, Projections, and Least Squares](1.%20Linear%20Algebra/03_vector_spaces_projections.ipynb) — Connect linear independence and subspaces to projection, orthogonalization, and least-squares fitting.
- [Eigenvalues and Matrix Decompositions](1.%20Linear%20Algebra/04_eigenvalues_and_decompositions.ipynb) — Understand spectral structure and the decompositions commonly used in PCA, least squares, numerical solvers, and dimensionality reduction.
- [Matrix Calculus, Tensors, and Quadratic Forms](1.%20Linear%20Algebra/05_matrix_calculus_tensors_and_quadratic_forms.ipynb) — Connect matrix derivatives, covariance structure, higher-dimensional tensors, and einsum notation to deep-learning computations.

### 2. Calculus
- [Derivatives, Partial Derivatives, and the Chain Rule](2.%20Calculus/06_derivatives_and_chain_rule.ipynb) — Learn the derivatives that explain how model outputs and losses change when inputs or parameters change.
- [Gradients, Jacobians, VJPs, and Hessians](2.%20Calculus/07_gradients_jacobians_vjp_hessians.ipynb) — Understand the derivative objects and vector-Jacobian products used by automatic differentiation systems.
- [Integration, Taylor Approximation, and Calculus for ML](2.%20Calculus/08_integration_taylor_and_calculus_for_ml.ipynb) — Cover integrals as accumulation and expectation, plus Taylor approximation, convexity, and first- and second-order optimality.

### 3. Probability and Statistics
- [Probability, Conditional Probability, and Bayes’ Rule](3.%20Probability%20and%20Statistics/09_probability_bayes_and_random_variables.ipynb) — Build the probability foundations used to reason about uncertainty, predictions, and evidence.
- [Common Probability Distributions](3.%20Probability%20and%20Statistics/10_probability_distributions.ipynb) — Meet the discrete, continuous, and multivariate distributions that appear in likelihood models and generative methods.
- [Descriptive and Inferential Statistics](3.%20Probability%20and%20Statistics/11_descriptive_and_inferential_statistics.ipynb) — Summarize data, quantify variability, understand correlation, and distinguish descriptive estimates from statistical inference.
- [MLE, MAP, and Statistical Origins of Loss Functions](3.%20Probability%20and%20Statistics/12_mle_map_and_loss_functions.ipynb) — Derive common loss functions from probability models and connect Bayesian priors to regularization.

### 4. Numerical Computation
- [Floating-Point Arithmetic and Numerical Stability](4.%20Numerical%20Computation/13_floating_point_and_numerical_stability.ipynb) — Learn why mathematically equivalent formulas may behave differently on a computer and how to avoid common numerical failures.
- [Numerical Solvers, Finite Differences, and Automatic Differentiation](4.%20Numerical%20Computation/14_numerical_solvers_finite_differences_autodiff.ipynb) — Compare iterative numerical methods with finite differences and automatic differentiation as used in modern deep learning.

### 5. Optimization
- [Gradient Descent and Newton Methods](5.%20Optimization/15_gradient_descent_and_newton_methods.ipynb) — Learn optimization objectives, gradient descent updates, learning-rate effects, and the role of second-order curvature.
- [Constrained Optimization, Convexity, Lagrange Multipliers, and KKT](5.%20Optimization/16_constrained_and_convex_optimization.ipynb) — Explore feasible regions, convex functions, Lagrange multipliers, and the KKT conditions behind constrained optimization.
- [SGD, Momentum, RMSprop, Adam, AdamW, and Loss Landscapes](5.%20Optimization/17_sgd_momentum_rmsprop_adam_adamw.ipynb) — Compare practical first-order optimizers and understand how loss geometry can complicate deep-network training.

### 6. Information Theory
- [Entropy, Cross-Entropy, KL Divergence, JS Divergence, and Mutual Information](6.%20Information%20Theory/18_entropy_cross_entropy_kl_js_mutual_information.ipynb) — Learn the information-theoretic quantities underlying classification losses, distribution comparison, representation learning, and GAN objectives.
- [Rate-Distortion, Compression, and the Information Bottleneck](6.%20Information%20Theory/19_rate_distortion_and_information_bottleneck.ipynb) — Relate information to lossy compression, representation learning, and the trade-off between compactness and predictive usefulness.

### 7. Advanced Topics
- [Stochastic Processes, Monte Carlo, Importance Sampling, and MCMC](7.%20Advanced%20Topics/20_stochastic_processes_sampling_mcmc.ipynb) — Introduce sequences of random variables, Markov chains, Monte Carlo estimates, importance sampling, and MCMC diagnostics.
- [Graph Theory, Graph Laplacians, and Graph Neural Networks](7.%20Advanced%20Topics/21_graph_theory_laplacian_and_gnns.ipynb) — Connect adjacency matrices, degrees, Laplacians, and message-passing operations used in graph learning.
- [Differential Geometry and Topology for Geometric Learning](7.%20Advanced%20Topics/22_differential_geometry_and_topology.ipynb) — Build intuition for manifolds, tangent directions, geodesics, curvature, and topology without requiring advanced geometric machinery.
- [ODEs, Dynamical Systems, and Neural ODEs](7.%20Advanced%20Topics/23_odes_and_dynamical_systems_neural_odes.ipynb) — Connect differential equations with state evolution, numerical integration, recurrent models, and continuous-depth neural networks.
- [Harmonic Analysis, Fourier Transforms, and Convolution](7.%20Advanced%20Topics/24_harmonic_analysis_fourier_and_convolutions.ipynb) — Understand frequency-domain representations and why Fourier analysis is useful for signals, convolution, and CNN intuition.

## Suggested learning order

1. Linear algebra: vectors, matrices, projections, decompositions, matrix calculus, and tensors.
2. Calculus: derivatives, the chain rule, Jacobians/VJPs, integration, and Taylor approximation.
3. Probability and statistics: Bayes' rule, distributions, inference, and MLE/MAP.
4. Numerical computation: floating-point stability, solvers, finite differences, and automatic differentiation.
5. Optimization: gradient-based methods, constraints, optimizers, and loss landscapes.
6. Information theory: entropy, cross-entropy, KL/JS divergence, mutual information, and rate-distortion.
7. Advanced topics: stochastic processes, graph learning, differential geometry, ODEs, and Fourier analysis.

## Notebook conventions

- Explanations and code comments are in English.
- Each notebook contains objectives, short examples, edge cases or practical cautions, exercises, and suggested solutions.
- Most code cells are self-contained; a notebook should be run from top to bottom.
- Results can vary slightly across hardware and library versions because floating-point arithmetic and random sampling are approximate. Random examples use seeded generators when reproducibility matters.

## Important note

These notes are a learning resource, not a substitute for checking assumptions in a real application. Numerical outputs should be interpreted in context, especially for statistical inference, optimization convergence, and scientific computing.
