---
title: "Code"
permalink: /code/
author_profile: true
redirect_from:
  - /code.html
---

{% include base_path %}

### Toolboxes
- **semgsim**, an R package that generates realistic surface electromyographic (sEMG) and force signals by modeling muscle physiology, and motor unit pool organization. [[Link]](https://github.com/ime-luebeck/semgsim).
- A **cardiac artifact removal toolnox** that provides Matlab implementations of a number of algorithms for removing cardiac interference from surface EMG measurements, as well as metrics and two exemplary datasets for evaluating their respective accuracy. [[Link]](https://github.com/e-pet/ecg-removal).
- **KFS-suite**, a collection of linear and nonlinear Kalman filter and smoother implementations in Matlab. The implementation can deal with missing data, multiple measurements, time-varying systems, and state constraints (a simple projection-based approach is implemented). A state-of-the-art iterated approximation scheme is implemented for nonlinear systems. [[Link]](https://github.com/e-pet/kfs_suite).

### Matlab utilities
- **scatter_nice**, a wrapper for Matlab's `scatter` function that actually generates useful plots by default. [[Link]](https://github.com/e-pet/scatter_nice).
- **plot_signals**, a simple method for conveniently plotting and comparing a bunch of signals in a matrix. [[Link]](https://github.com/e-pet/plot_signals).
- **mvdensity**, an efficient multivariate density estimation method based on simple histogram interpolation/smoothing. [[Link]](https://github.com/e-pet/mvdensity).
- **movquant**, the equivalent of `medfilt1`, `movmax`, and `movmin`, but for arbitrary quantiles p. [[Link]](https://www.mathworks.com/matlabcentral/fileexchange/84200-movquant/).
- **gen_rand_spd**, a simple function to generate a random symmetric positive definite matrix with a specified condition number. [[Link]](https://github.com/e-pet/gen_rand_spd).

### Guidance
- A simple latex template for Bachelor's / Master's theses. [[Link]](https://github.com/e-pet/thesis-template).
- An introductory document on best practices for scientific software development. [[Link]](https://github.com/e-pet/best-practices-scientific-software-dev/blob/master/best_practices_scientific_software_dev.md).

