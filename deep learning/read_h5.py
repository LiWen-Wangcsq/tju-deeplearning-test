import h5py
import numpy as np

p = '/data1/0_Brain_Seg/WLW/unet/data/BraTS2021/dataset/BraTS2021_00329_mri_norm2.h5'
h5f = h5py.File(p, 'r')
image = h5f['image'][:]
label = h5f['label'][:]

print('image shape:',image.shape,'\t','label shape',label.shape)
print('label set:',np.unique(label))
