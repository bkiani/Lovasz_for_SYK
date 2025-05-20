# Commutation Graph Lovász Number Calculator

  

A small toolkit for constructing commutation graphs from $k$-subsets of $n$ Fermions, computing the Lovász theta number, and aggregating results over randomized sparsified graphs.

  

  

  

## Paper

  

* Preprint: [arXiv\:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

  


  

## Overview

  

This repository contains two main scripts:

  

1. **`lovasz.py`**: Helper module for parsing graphs and computing the Lovász theta number of a commutation graph via semidefinite programming (SDP).

  

2. **`save_outputs.py`**: Runs multiple trials to sparsify large commutation graphs and record their Lovász theta numbers.




  

## Dependencies

  

* Python >= 3.7

* **CVXPY** for formulating and solving SDPs

* **MOSEK** *(other solvers such as SCS are also supported; see CVXPY documentation)*

* **NetworkX** for graph construction

* **NumPy** for array operations

* **SciPy** (for combinatorial functions)

* **Pandas** for data handling and CSV export

  



## Usage

  

1. **Compute Lovász number for a small graph**

  

```bash

python lovasz.py

```

  

2. **Aggregate results over trials**

  

```bash

python save_outputs.py

# Results will be stored in results/output_results_agg.csv

```

  

Feel free to modify solver options in `lovasz_theta` or parameters in `save_outputs.py` to suit your experiments.


