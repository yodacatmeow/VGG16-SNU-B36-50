"""
"tsne_input.py"

References
    # tSNE
        https://github.com/bwcho75/dataanalyticsandML/blob/master/dimension%20reduction/2.%20t-SNE%20visualization.ipynb

"""
# Import public Python modules
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.manifold import TSNE


# Custom Python modules
from gen_data import generate
from load_data import load


## TDDO
learning_rate = 10                                                  # Learning rate
gen_data = True                                                    # Generate dataset (.p format)
run_tsne = True                                                    # Run tSNE
save_tsne = True                                                   # save tSNE results
fold = 1                                                            # Which fold is set as input
rec_name = 'result/tsne_input.csv'                                  # save tSNE results as "rec_name".csv
metadata_path = 'dataset/metadata_5fcv_box.csv'                     # Where is meta-data?
traindata_path = 'dataset/train_5fcv_k' + str(fold) + '.p'          # Where is input data (.p format)
label_column_name = 'category'                                      # Which row represents 'category'?

# Batch_size; In this case, batch size need to be '1'
batch_size_tr = 1

# If "gen_data" = True, generate dataset in .p format
if gen_data:
    generate(metadata_path=metadata_path, data_path=traindata_path,
             batch_size=batch_size_tr, label_column_name=label_column_name,
             is_training=True, fold=fold)
else:
    pass

# Load Mel-spectrograms and flatten them into a vector
dataframe = load(traindata_path, batch_size_tr)
feature = []
label = []
for i, row in dataframe.dataframe.iterrows():
    patch = row['patch'][:,:,0]
    #- Reshaping
    patch = np.reshape(patch, [int(patch.shape[0] * patch.shape[1])])
    category = row['category']
    #print("shape of tid:", patch.shape, "category:", category)
    feature.append(patch)
    label.append(category)
print("feature appending completed")

# Initialize "xs" and "ys"
xs = []
ys = []
# tSNE
if run_tsne:
    model = TSNE(learning_rate=learning_rate)
    transformed = model.fit_transform(feature)
    xs = transformed[:,0]
    ys = transformed[:,1]
    marker = label
else:
    # Read  csv file
    df = pd.read_csv(rec_name, sep=',')
    xs = df['xs']
    ys = df['ys']
    marker = df['label']

# Save results
if save_tsne:
    record = pd.DataFrame()
    record['xs'] = xs
    record['ys'] = ys
    record['label'] = label
    record.to_csv(rec_name)
else:
    pass

# Draw a scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xs, ys, c=marker)

for i, txt in enumerate(marker):
    ax.annotate(txt, (xs[i], ys[i]))

plt.show()


