# Vectorised Skills Space Method (VSSM)

#### An Implementation of Skill Set Similarity using Matrix Operations

## What is this Project?

The Skills Space Method developed in [“Skill-driven Recommendations for Job Transition Pathways” (Dawson, Williams & Rizoiu 2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254722) is a novel approach to measuring the distance between sets of skills. Sets of skills in the context of this report are only associated with occupations (job titles), but can be generalised to other groups such as company or industry.

[(Dawson, Williams & Rizoiu 2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254722) provide the mathematical foundation for computing the Skill Set Similarity between two sets. However, practically implementing the functions and computations is not documented. A naive approach, that mimics the exact mathematical outline (e.g. using for loops to sum, lazy functions) is not feasible on large datasets due to poor execution performance.

As such, this report outlines and explains a method for implementing the Skills Space Method using matrix operations. The vectorisation of this method enables faster execution through SIMD or parallel computing (depending on the platform) and memoisation of necessary, static values in matrix form.

## Project Dependencies and Setup


## Example VSSM Usage


## Project Structure


## Testing

- Run all tests: `python -m unittest discover -s tests -p "*_test.py"`
- Run a specific test: `python -m unittest tests.<test-module>.<test-class>.<test-method>`
