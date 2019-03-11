# VGG16-SNU-B36-50
- This project is for classification of inter-floor noise ([SNU-B36-50](https://github.com/yodacatmeow/SNU-B36-50)) in a building using VGG16
- VGG16 is fine-tuned on [SNU-B36-50](https://github.com/yodacatmeow/SNU-B36-50) without freezing any weights
- The model is evaluated using 5-fold cross-validation
- The following confusion matrix shows the evaluation results


![](https://github.com/yodacatmeow/VGG16-SNU-B36-50/blob/master/figure/cfmtx2.png)



## Notice

- **VGG16-SNU-B36-50** will be merged into **IndoorNoise** repository in near future




## Requirements

- Python (version 3.5.2)
- Python modules : TensorFlow (version 1.2), Numpy, Scipy, Pandas, matplotlib, librosa, and Pickle
- [Pretrained weights of VGG16](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)



## Contents

- [audio](https://github.com/yodacatmeow/VGG16-SNU-B36-50/tree/master/audio): This folder includes inter-floor noises ([SNU-B36-50](https://github.com/yodacatmeow/SNU-B36-50)) for training and validation
- [dataset](https://github.com/yodacatmeow/VGG16-SNU-B36-50/tree/master/dataset): This folder includes metadata. Also, when the audio clips are converted to log scaled Mel-spectrograms they are saved into this folder


- [result](https://github.com/yodacatmeow/VGG16-SNU-B36-50/tree/master/result): Cross-validation accuracy and confusion matrix are saved into this folder
- [cfmtx.py](https://github.com/yodacatmeow/VGG16-SNU-B36-50/blob/master/cfmtx.py): This includes a confusion matrix drawing function
- [feature.py](https://github.com/yodacatmeow/VGG16-SNU-B36-50/blob/master/feature.py): This includes ```melspec2``` a function which converts the audio clips to log scaled Mel-spectrograms using [LibROSA](https://librosa.github.io/librosa/)
- [gen_data.py](https://github.com/yodacatmeow/VGG16-SNU-B36-50/blob/master/gen_data.py): This reads the metadata and the audio clips are converted to log scaled Mel-spectrograms using **feature.py**.  The Mel-spectrograms are saved as **.p** and saved in "dataset"
- [load_data.py](https://github.com/yodacatmeow/VGG16-SNU-B36-50/blob/master/load_data.py): This can load training data and validation data (Currently, this supports batch and mini-batch)
- [vgg16_adap.py](https://github.com/yodacatmeow/VGG16-SNU-B36-50/blob/master/vgg16_adap.py): This builds network architecture of VGG16 with an adaptation layer. Also, this supports ```save_weights()``` which saves weights as **.npz** after a training
- [main.py](https://github.com/yodacatmeow/VGG16-SNU-B36-50/blob/master/main.py): You need to set several parameters. The current settings in the code are for *IWAENC 2018* submission
  - ```gpu_device``` : Select a gpu device
  - ``is_transfer_learn`` : Transfer the pretrained weights or not
  - ```gen_data```: If this is TRUE, then the audio clips are converted to log scaled Mel-spectrograms and they are saved as **.p**
  - ```freeze_layer``` 
    - True: Freeze the weights except **fc3w**, **fc3b**, **fc4w**, and **fc4b**
    - False: Do not freeze all the weights
  - ```bn```: If this is TRUE,  turn on batch normalization
  - ```saver```: If it is TURE, the weights at the last epoch are saved as **.npz**
  - ```fold```: Use *k*-th subsample as the validation set
- [results](https://github.com/yodacatmeow/VGG16-SNU-B36-50/tree/master/result): Training loss, validation loss, training accuracy, and validation accuracy are saved to here




## Quick start

- Clone this project ```git clone https://github.com/yodacatmeow/VGG16-SNU-B36-50```

- Download the [pretrained weights](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) to this project path
- Start a process  ```CUDA_VISIBLE_DEVICE=0 python3 main.py``` 




## Citing

```
@inproceedings{choi2018floornoise,
  title={Classification of noise between floors in a building using pre-trained deep convolutional neural networks},
  author={Choi, Hwiyong and Lee, Seungjun and Yang, Haesang and Seong, Woojae},
  booktitle={2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC)},
  pages={535--539},
  year={2018},
  organization={IEEE}
}
```

