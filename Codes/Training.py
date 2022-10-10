"""
Main script for network training and testing
Definition of the command-line arguments are in model.py and can be displayed by `python Training.py -h`

Author: Ting Gong 
"""

import numpy as np
import os
import time

from scipy.io import savemat
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                                            EarlyStopping

from utils import save_nii_image, calc_RMSE, loss_func, repack_pred_label, \
                  MRIModel, parser, load_nii_image, unmask_nii_data, loss_funcs, fetch_train_data_MultiSubject, \
                  set_randomness

# Get parameter from command-line input
args = parser().parse_args()

train_subjects = args.train_subjects
#test_subject = args.test_subject[0]
nDWI = args.DWI
scheme = "first"
mtype = args.model
train = args.train
out_path = args.out

# Get random seed
rseed = args.rseed
# Set random seed
if rseed is not None:
    set_randomness(rseed)

# define output path for tensorboard
if out_path is not None:
    tb_out = os.path.join(out_path,'logs')
    if not os.path.isdir(out_path):
        os.system("mkdir " + out_path)
else:
    tb_out = 'logs'

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

# Parameter name definition
savename = str(nDWI)+ '-'  + scheme + '-' + args.model

# Constants
types = ['NDI' , 'FWF', 'ODI']
ntypes = len(types)
decay = 0.1

# do we want to shuffle the data before train/validation split?
data_shuffle = args.data_shuffle

shuffle = args.train_shuffle
patience = args.patience
y_accuracy = None
output_accuracy = None
y_loss = None
output_loss = None
nepoch = None

# Define the adam optimizer
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Train on the training data.
if train:
    # Define the model.
    model = MRIModel(nDWI, model=mtype, layer=layer, train=train, kernels=kernels)

    model.model(adam, loss_funcs[loss], patch_size)

    # get the training data
    if rseed is not None:
        set_randomness(rseed + 1)
    data, label = fetch_train_data_MultiSubject(train_subjects, mtype, nDWI, scheme, data_shuffle)


    # Reduce learning rate when a metric has stopped improving.
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=patience, epsilon=0.0001)
    tensorboard = TensorBoard(log_dir=tb_out, histogram_freq=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.0000005)

    if rseed is not None:
        set_randomness(rseed+2)
    [nepoch, output_loss, y_loss, output_accuracy, y_accuracy] = model.train(data, label, batch_size, epochs,
                                                                   [reduce_lr, tensorboard, early_stop],
                                                                   savename, out_path, shuffle=shuffle,
                                                                   validation_data=None)
