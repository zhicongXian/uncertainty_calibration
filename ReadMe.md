# Temperature Scaling for Probability Calibration of Deep Learning Methods

## Paper

This is a numpy implementation of the original proposed method: [on Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

It preserves the model prediction accuracy and scales the logits for different classes by the same parameter
called temperature. The key founding from the paper is that probability calibration for deep learning method is only a 
low-dimensional task. 

## Additional tool

For the visualization of the reliability diagram, I used the implementation from this [github](https://github.com/hollance/reliability-diagrams)

## Usage

To start with, refer to the examples in main.py

```bash
python main.py
```

## Contribution

- The original [implementation](https://github.com/gpleiss/temperature_scaling) in the paper is in pytorch. A lot of deep
learning methods are also implemented using tensorflow, keras, Jax or FAX. Therefore, this implementation using numpy 
can also be applied to calibrate models implemented in other deep learning framework. 

- The optimizer employed in the original implementation is BFGS, which is a gradient-based method. Since the logits are
scaled by `1/T` with `T` as the temperature parameter, when it approaches to zero and can lead to numerical instability. 
We proposed to use [Nelder-Mead](https://link.springer.com/article/10.1007/s10589-010-9329-3) optimizer, which is a simplex
optimization method and non-gradient based. Henceforth, better numerical instability can be achieve. 




