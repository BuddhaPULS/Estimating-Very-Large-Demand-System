
# Large-Scale Demand System Estimation with JAX

A scalable framework for estimating large-scale demand systems using JAX for accelerated computation.

This project implements a discrete choice random utility model and introduces efficient estimation techniques for consumer demand across a large number of products.

## Motivation

Estimating demand systems with thousands of products is computationally expensive due to the large choice space.

This project addresses:
- Scalability challenges in discrete choice models
- High-performance estimation using JAX

## Methodology

The estimation procedure combines:
- Stochastic gradient ascent for scalable optimization
- Random choice set sampling (McFadden-style)
- Method of simulated scores

## Data

The dataset consists of:
- `q.parquet`: consumption (quantity) data
- `p.parquet`: price data

## Computation Optimization

The core algorithm is refactored using JAX to enable high-performance computation:

- Vectorized matrix operations
- JIT compilation with operator fusion
- Removal of dynamic control flow

This reduces computation time by approximately **5×**.

## Results

The best-performing model uses:
- 75% BNTR propagation
- 25% simulated score contribution

It achieves an average absolute prediction error of approximately **1.5**.

## Tech Stack

- Python
- JAX
- Pandas / NumPy
