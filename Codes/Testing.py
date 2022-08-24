"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python Testing.py -h`

Author: Ting Gong 
"""

import numpy as np
import os
import time

from scipy.io import savemat, loadmat

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                                            EarlyStopping

from utils import save_nii_image, calc_RMSE, loss_func, repack_pred_label, \
                  MRIModel, parser, load_nii_image, unmask_nii_data, loss_funcs, fetch_train_data_MultiSubject


# Get parameter from command-line input
args = parser().parse_args()

train_subjects = args.train_subjects
test_subject = args.test_subject[0]
nDWI = args.DWI
scheme = "first"
mtype = args.model
out_path = args.out

# determin the input volumes using a scheme file
combine = None
schemefile = args.scheme
if schemefile is not None:
    combine = np.loadtxt('schemes/' + schemefile)
    combine = combine.astype(int)
    nDWI = combine.sum()
    scheme = schemefile

lr = args.lr
epochs = args.epoch
kernels = args.kernels
layer = args.layer

loss = args.loss
batch_size = args.batch
patch_size = args.patch_size
label_size = patch_size - 2
base = args.base

# Constants
types = ['NDI' , 'FWF', 'ODI']
ntypes = len(types)
decay = 0.1

# Parameter name definition
savename = str(nDWI)+ '-'  + scheme + '-' + args.model

# Define the adam optimizer
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Load testing data
mask = load_nii_image('datasets/mask/mask_' + test_subject + '.nii')
tdata = loadmat('datasets/data/' + test_subject + '-' + str(nDWI) + '-' + scheme + '.mat')['data']

# Reshape the data to suit the model.
if mtype[:6] == 'conv3d':
  tdata = np.expand_dims(tdata, axis=0)
elif mtype[:6] == 'conv2d':
  tdata = tdata.transpose((2, 0, 1, 3))

test_shape = args.test_shape
if test_shape is None:
  test_shape = tdata.shape[1:4]

# Define the model
model = MRIModel(nDWI, model=mtype, layer=layer, train=False, kernels=kernels, test_shape=test_shape)
model.model(adam, loss_func, patch_size)
model.load_weight(savename, out_path)

weights = model._model.layers[1].get_weights()

# Predict on the test data.
time1 = time.time()
pred = model.predict(tdata)
time2 = time.time()

time3 = time.time()
pred = repack_pred_label(pred, mask, mtype, ntypes)
time4 = time.time()

#print "predict done", time2 - time1, time4 - time3

# Save estimated measures to /nii folder as nii image
# os.system("mkdir -p nii")
# Modified for windows system
if out_path is not None:
    nii_out = os.path.join(out_path, 'nii')
else:
    nii_out = 'nii'
if not os.path.isdir(nii_out):
    os.system("mkdir " + nii_out)

for i in range(ntypes):
    data = pred[..., i]
    if out_path is not None:
        filename = os.path.join(nii_out, test_subject + '-' + types[i] + '-' + savename + '.nii')
    else:
        filename = os.path.join(nii_out, test_subject + '-' + types[i] + '-' + savename + '.nii')

    data[mask == 0] = 0
    save_nii_image(filename, data, 'datasets/mask/mask_' + test_subject + '.nii', None)
