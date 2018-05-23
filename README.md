# VGG16-SNUB36-50
- This project is for source type and source location classification of noise between floors (NBF) in a building using [pre-trained](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) VGG16
- An adaptation layer is added to VGG16 for dimension adaption
- VGG16 is fine-tuned on [SNU-B36-50](https://github.com/yodacatmeow/SNU-B36-50) for 50 epochs without freezing any layers. Details of the dataset are explained in [here](https://github.com/yodacatmeow/SNU-B36-50)
- The model is evaluated via cross-validation





## Requirements

- Python (developed and test with version 3.5.2)
- Python modules : TensorFlow (developed and tested with version 1.2), Numpy, Scipy, Pandas, matplotlib, librosa, and Pickle
- [Pretrained weights of VGG16](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) : It need to be located in the project path





## Contents

- "audio": This folder includes audio clips ([SNU-B36-50](https://github.com/yodacatmeow/SNU-B36-50)) for training and validation
- "dataset": This folder includes metadata. Also, when the audio clips are converted to log scaled Mel-spectrograms, they will be saved into this folder


- "result": Cross-validation accuracy and confusion matrix are saved into this folder
- "cfmtx.py": This includes functions for building  and drawing the confusion matrix
- "feature.py": It include ```melspec2``` a function which converts the audio clips to log scaled Mel-spectrograms using [LibROSA](https://librosa.github.io/librosa/)
- "gen_data.py": This reads the metadata and by referring it, the audio clips are converted to log scaled Mel-spectrograms using "feature.py".  The Mel-spectrograms are saved as **.p** and located in "dataset"
- "load_data.py": This can load training data and validation data (Currently, this supports batch and mini-batch)
- "vgg16_adap.py": This builds network architecture of VGG16 with the adapation layer. Also, this supports ```save_weights()``` which saves weights as **.npz** after training
- "main.py": You need to set several parameters. The current settings in the code are for *IWAENC 2018* submission
  - ```gpu_device``` : Select a gpu device
  - ``is_transfer_learn`` : Transfer the pretrained weights or not?
  - ```gen_data```: If this is TRUE, then the audio clips are converted to log scaled Mel-spectrograms and they are saved as **.p**
  - ```freeze_layer``` 
    - True: Freeze the layers except **fc3w**, **fc3b**, **fc4w**, and **fc4b**
    - False: Do not freeze the layers
  - ```bn```: If this isTRUE,  then turn on batch normalization
  - ```saver```: If it is TURE, the weights at the last epoch are saved as **.npz**
  - ```fold```: Use *k*-th subsample as the validation set
- "results": Training loss, validation loss, training accuracy, and validation accuracy are saved to here
- "test.py": Not used for this work
- "tsne_input": *TBU*
- "tsne_conv": *TBU*




## Quick start

- Clone this project ```git clone https://github.com/yodacatmeow/VGG16-SNU-B36-50```

- [Download](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) the pretrained weights to this project path
- Start a process  ```CUDA_VISIBLE_DEVICE=0 python3 main.py``` 

  - If you don't prepare TensorFlow GPU version, set ```gpu_device``` in "main.py" as ```gpu_device = 'device:cpu:0'``` and start a process ```python3 main.py```




## Citing

Hwiyong Choi, Seungjun Lee, Haesang Yang, and Woojae Seong (2018). *Classification of Noise between Floors in a Building Using Pre-trained Deep Convolutional Neural Networks*.  Submitted for *Acoustic Signal Enhancement (IWAENC), 2018 IEEE International Workshop on*. IEEE.



## License

[License](https://github.com/yodacatmeow/VGG16_SNUB36-50/blob/master/LICENSE)
