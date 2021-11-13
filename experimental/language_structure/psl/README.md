# Tensorflow with Differentiable Constraints

This codebase contains code that executes tensorflow programs using
differentiable constraints.

**Inference:**

- Gradient Decoding
- Beam Search Decoding

**Learning:**

- Constraint Regularization

**Models**

- [MultiWoZ Synthetic](https://github.com/stanford-oval/zero-shot-multiwoz-acl2020)

## Requirements

**System Requirements**

`python>=2.7.0` <br />
`tensorflow>=2.5.1` <br />
`numpy>=1.19.5`

*Install Python Requirements*

`pip install -r requirements.txt`

## Running

**MultiWoZ Synthetic Gradient Decoding:**

`python3 -m scripts.multiwoz_synthetic.gradient_decoding_run path/to/data`