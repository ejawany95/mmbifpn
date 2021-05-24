import os
from sklearn.model_selection import StratifiedKFold
import numpy as np

root = '/media/data1/suriza/MICCAI_BraTS2020_TrainingData_2'
# valid_data_dir = '/media/data1/suriza/research/BraTS-DMFNet/MICCAI_BraTS_2018_Data_Validation'

def write(data, fname, root=root):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))

hgg = os.listdir(os.path.join(root, 'HGG'))
hgg = [os.path.join('HGG', f) for f in hgg]

lgg = os.listdir(os.path.join(root, 'LGG'))
lgg = [os.path.join('LGG', f) for f in lgg]

X = hgg + lgg
Y = [1]*len(hgg) + [0]*len(lgg)

write(X, 'all.txt')

X, Y = np.array(X), np.array(Y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

for k, (train_index, valid_index) in enumerate(skf.split(Y, Y)):
    train_list = list(X[train_index])
    valid_list = list(X[valid_index])

    write(train_list, 'train_20_{}.txt'.format(k))
    write(valid_list, 'valid_20_{}.txt'.format(k))

# valid = os.listdir(os.path.join(valid_data_dir))
# valid = [f for f in valid if not (f.endswith('.csv') or f.endswith('.txt'))]
# write(valid, 'valid.txt', root=valid_data_dir)