# Training
- Start learning from the pretrained weights
- Train all the layers (do not freeze)
- Gradient descent (minibatch size = 39)
- Learning rate = 0.001
- Epoch = 100





# Input feature

- Dimension: 224 x 224 x 3
  - Ch0: MFCC with log scale
  - Ch1: MFCC with log scale
  - Ch2: MFCC with log scale
- MFCC with log scale
  - 3.0 sec audio, fs = 44,100 Hz
  - Size of the patch is fixed to 224 x 224
  - Window size = 2048, nFFT = 2048, hop size = 592
  - Audio frequency range: 0 ~ fs/2



# Validation Accuracy

|   Fold *k*   |  1   |  2   |  3   |  4   |  5   | Mean |
| :----------: | :--: | :--: | :--: | :--: | :--: | :--: |
| Accuracy (%) | 81.5 | 88.7 | 84.9 | 84.9 | 83.8 | 84.8 |