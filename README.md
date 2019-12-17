# README

This repository holds the code for the Keras layers implementation of Behler's Atom-Centered Symmetry Function, in the context of the following paper:

```
Profitt, T.A. and Pearson, J.K., 2019. A shared-weight neural network architecture for predicting molecular properties. Physical Chemistry Chemical Physics, 21(47), pp.26175-26183.
```

## Installation

The easiest way to use the layers is to install them as the package using the normal package management methods for Python. The layers are not yet released as a Python package under PyPI, so one will have to clone this repository and install using the files themselves.

```bash
git clone https://github.com/UPEIChemistry/atomic-images.git
```

Then install by pointing `pip` to the directory:
```bash
pip install /path/to/atomic-images/
```

Once installed, the layers should be importable as normal:
```python
from atomic_images import layers
```

## Usage

In the `examples` directory, one can find examples of using the layers for both numpy operations as well as Keras models. Consult the source code and docstrings in `atomic_images/layers.py` for more information regarding input and output shapes and parameters.

I intend to write a full script to demonstrate usage on the QM9 in the future, but in the current state of the repository, the layers should be usable to build ASCF models in Keras.

## Contributions

Contributions are welcome. You can fork the repository and submit a pull request to have documentation or code changed. Submit issues or pull requests for reporting bugs or discussing potential changes.
