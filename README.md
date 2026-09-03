# Schrödinger Equation Solver with Wood-Saxon Potential

## Overview

This program solves the radial Schrödinger equation with a Wood-Saxon potential using numerical methods. The implementation employs the finite difference method to discretize and solve the differential equation efficiently.

## Method

### Finite Difference Method
The finite difference method is a numerical approach for solving differential equations by:
- Discretizing the domain into a grid of points
- Approximating derivatives using finite differences
- Converting the differential equation into a system of linear equations
- Solving the resulting eigenvalue problem

### Wood-Saxon Potential
The Wood-Saxon potential is a nuclear potential commonly used in quantum mechanics:
- Models the nuclear mean-field potential
- Used in nuclear structure calculations
- Characterized by smooth diffuse surface and saturated interior

## Features

- Numerical solution of the radial Schrödinger equation
- Wood-Saxon potential implementation
- Eigenvalue and eigenfunction computation
- Visualization of results

## Usage

```bash
# Clone the repository
git clone https://github.com/poledtomas/Woods-Saxon-potential.git

# Navigate to the directory
cd Woods-Saxon-potential

# Run the program
python solve_schrodinger.py
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- SciPy

## Installation

```bash
pip install numpy matplotlib scipy
```

## Results

The program outputs:
- Eigenvalues (energy levels)
- Eigenfunctions (wave functions)
- Plots comparing numerical and analytical solutions (if available)

## References

- [Schrödinger Equation](https://en.wikipedia.org/wiki/Schrödinger_equation)
- [Wood-Saxon Potential](https://en.wikipedia.org/wiki/Woods-Saxon_potential)
- Numerical Methods in Quantum Mechanics
