# EvoMoE architecture

EvoMoE is the network inside AstroVerse. It reads one phase-folded light curve and twelve stellar columns, and it returns two class logits plus the three weights that produced them.

The implementation is `research/evonex/src/model_evonex.py`.

| | |
| --- | --- |
| Flux input | 2,000 samples |
| Stellar input | 12 TIC columns, native catalog scales |
| Embedding | 128 |
| Parameters | 1,485,605, all trainable |
| Output | 2 logits and a gate in R³ |
| Optimizer | Adam, cross-entropy. The catalog schedule is 100 epochs, batch 256, learning rate 10⁻⁴, weight decay 10⁻⁵, gradient clip 1, cosine decay, seed 42 |

Training does not search a population. EvoMoE is the name of the mixture.

## Forward pass

Each expert returns an embedding h and a scalar confidence c.

```
g = softmax(c)                         three weights, sum to 1
h = g_cnn h_cnn + g_trans h_trans + g_phys h_phys
z = Linear(64 → 2)(ReLU(Dropout(Linear(128 → 64)(h))))
```

Dropout on the shared head is 0.3 during training. The call returns z and g.

An expert can be removed from one pass by replacing its confidence with negative infinity before the softmax. Its parameters stay allocated.

## Convolutional expert — local shape

Three parallel Conv1d maps, 1 → 16 channels, kernels 5, 11, and 21, padding 2, 5, and 10, then ReLU. The maps are concatenated to 48 channels. Max-pool of width 4, a convolution 48 → 64 with kernel 5 and ReLU, and a second max-pool of width 4 leave a length of 125. Flattening gives 8,000 units, projected to 128. A linear map 128 → 1 is the confidence.

This block holds 1,040,321 parameters. 1,024,128 of them are the map from 8,000 to 128.

## Transformer expert — the folded sequence

The series is cut into 100 patches of length 20. Each patch is projected 20 → 32. Sinusoidal positions are added. Two encoder layers follow, 4 heads, feed-forward width 64, ReLU, dropout 0.1. The tokens are flattened to 3,200 and projected to 128, with the same confidence head.

This block holds 427,617 parameters. 409,728 of them are the map from 3,200 to 128.

## Physics expert — the star

A multilayer perceptron on the twelve TIC columns: 12 → 64, ReLU, dropout 0.2, then 64 → 128, ReLU, and a confidence head. It never sees the flux. It holds 9,281 parameters.

The columns, in order, are Tmag, Teff, logg, rad, mass, rho, lum, d, ebv, ra, dec, contratio.

## Where the time goes

On the CPU used to measure the reference module (Windows 11, Intel64 Family 6 Model 183, PyTorch 2.12.1, batch size 1, 100 calls after 20 warm-up calls) the median forward pass is 7.64 ms. The 5th and 95th percentiles are 6.67 ms and 10.36 ms.

## Reduction in front of the network

`process_lightcurve` keeps finite samples, prefers PDCSAP flux, divides by the median, removes a Savitzky–Golay trend (window 151, polynomial order 3), and sigma-clips at 5σ for at most five iterations.

Box least squares then searches 10,000 periods from 0.5 to 20 days and 10 durations from 0.01 to 0.2 days. The fold is

```
phase = ((t - t0 + 0.5 P) mod P) - 0.5 P
```

and the folded flux is linearly resampled to 2,000 samples.
