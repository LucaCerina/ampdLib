README.md rev. 10 Feb 2023 by Luca Cerina.
Copyright (c) 2023 Luca Cerina.
Distributed under the Apache 2.0 License in the accompanying file LICENSE.

# Automatic Multiscale-based Peak Detection (AMPD)

ampdLib implements automatic multiscale-based peak detection (AMPD) algorithm
as in An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and
Quasi-Periodic Signals, by Felix Scholkmann, Jens Boss and Martin Wolf,
Algorithms 2012, 5, 588-603.

### Python required dependencies
- Python >= 3.6
- Numpy
- Scipy for tests

### Installation
The library can be easily installed with setuptools support using `pip install .` or via PyPI with `pip install ampdlib`

### Usage
A simple example is:
```
peaks = ampdlib.ampd(input)
```

AMPD may require a lot of memory (N*(lsm_limit*N/2) bytes for a given length N and default lsm_limit). A solution is to divide the signal in windows with `ampd_fast` or `ampd_fast_sub` or determine a better lsm_limit for the minimum distance between peaks required by the use case with `get_optimal_size`. 

### Tests
The tests folder contains an ECG signal with annotated peaks in matlab format.

#### Contribution
If you feel generous and want to show some extra appreciation:

[![Buy me a coffee][buymeacoffee-shield]][buymeacoffee]

[buymeacoffee]: https://www.buymeacoffee.com/u2Vb3kO
[buymeacoffee-shield]: https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png
