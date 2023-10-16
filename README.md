# CircRL

A small library of mech interp tools, primarily focused on
interpreting RL policies (though most of the tools are general).

The library has three main components:
- **Hooks**: Tools for hooking into PyTorch models that provide
  simple, safe wrappers around PyTorch forward hooks
  functionality and allow easy caching, patching and arbitrary hook
  functions.
- **Probing**: Tools for training linear probes on model activations
  (or any other data), including sparse probes.
- **Rollouts**: Tools for running rollouts and collectiong various
  kinds of data through a unified interface.

## Installation

CircRL is available on PyPI, and can be installed with pip:

```bash
pip install circrl
```

## Usage

A detailed, self-contained demo of CircRL is available in the
[CircRL demo notebook](demo.ipynb).

## License

CircRL is licensed under the MIT license.

## Citation

If you use CircRL in your research, please cite according to:

```bibtex
@misc{circrl,
  author    = {MacDiarmid, Monte},
  title     = {CircRL},
  year      = {2023},
  url       = {https://github.com/montemac/circrl}
}
```
