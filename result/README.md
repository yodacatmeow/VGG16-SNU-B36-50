## Evaluation using 5-fold cross validation

- Folder name format = [**Transfer learning**] _ [**Freeze weights**] _ [**Input**] _ [**Window size**] _ [#**FFT points**] _ [**Hop size**] 
  - [**Transfer learning**]
    - **Blank**: w/o transfer learning
    - **tf**: w/ transfer learning
  - [**Freeze weights**]
    - f: freeze weights
    - nf: allow weight update
  - [**Input**]
    - mfccdb: log scaled MFCC
    - stftdb: log scaled spectrogram
    - mspdb: log scaled Mel-spectrogram



|        Model        | Fold 1 Accuracy (%) | Fold 2 Accuracy (%) | Fold 3 Accuracy (%) | Fold 4 Accuracy (%) | Fold 5 Accuracy (%) | Mean Accuracy (%) |
| :----------------------------: | :----: | :----: | :----: | :----: | :----: | :--: |
|     nf_mspdb_2048_2048_592     |  57.7  |  68.5  |  66.9  |  69.7  |  60.3  | 64.6 |
| tr_nf_mfccdb_2048_2048_592 |  81.5  |  88.7  |  84.9  |  84.9  |  83.8  | 84.8 |
|   tr_f_mspdb_2048_2048_592   |  89.2  |  94.6  |  92.8  |  93.8  |  91.0  | 92.3 |
| tr_nf_stftdb_2048_2048_592 |  90.0  |  97.2  |  95.1  |  96.2  |  94.9  | 94.7 |
| tr_nf_mspdb_2048_2048_592 |  96.4  |  97.9  |  96.4  |  96.7  |  96.2  | 96.7 |
| tr_nf_mspdb_2048_2048_296 |  95.4  |  98.7  |  98.7  |  97.4  |  97.7  | 97.6 |

