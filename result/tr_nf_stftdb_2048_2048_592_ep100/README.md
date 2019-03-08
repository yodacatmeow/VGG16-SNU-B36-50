# Training

- Start learning from the pretrained weights
- Train all the layers (do not freeze)
- Gradient descent (minibatch size = 39)
- Learning rate = 0.001
- Epoch = 100




# Input feature

- Dimension: 224x224x3

  - Ch0: Spectrogram (band0:band223)  with log scale
  - Ch1: Spectrogram (band224:band447)  with log scale
  - Ch2: Spectrogram (band448:band671)  with log scale

- Spectrogram
  - Size of the patch is fixed to 1024 x 224
  - Window size = 2048, nFFT = 2048, hop size = 592
  - Audio frequency range: 0 ~ fs/2




# Validation Accuracy

|   Fold *k*   |  1   |  2   |  3   |  4   |  5   | Mean |
| :----------: | :--: | :--: | :--: | :--: | :--: | :--: |
| Accuracy (%) | 90.0 | 97.2 | 95.1 | 96.2 | 94.9 | 94.7 |