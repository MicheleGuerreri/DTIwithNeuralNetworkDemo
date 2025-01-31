"""
Module definition for network training and testing.

Define your new model here

Author: Ting Gong
"""

import os
import argparse
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv3D, Flatten, Reshape, Conv2DTranspose, UpSampling2D, Concatenate


class MRIModel(object):
    """
    MRI models
    """

    _ndwi = 0
    _single = False
    _model = None
    _type = ''
    _loss = []
    _label = ''
    _kernel1 = 150
    _kernel2 = 150
    _kernel3 = 150

    def __init__(self, ndwi=96, model='fc1d', layer=3, train=True, kernels=None, ntypes=3, test_shape=[90, 90, 90]):
        self._ndwi = ndwi
        self._type = model
        self._hist = None
        self._train = train
        self._layer = layer
        self._ntypes = ntypes
        self._test_shape = test_shape
        if kernels is not None:
            self._kernel1, self._kernel2, self._kernel3 = kernels
   
    def _fc1d_model(self, patch_size):
        """
        Fully-connected 1d ANN model.
        """
        inputs = Input(shape=(self._ndwi,))
        # Define hidden layer
        hidden = Dense(self._kernel1, activation='relu')(inputs)
        for i in np.arange(self._layer  - 1):
            hidden = Dense(self._kernel1, activation='relu')(hidden)

        hidden = Dropout(0.1)(hidden)

        # Define output layer
        outputs = Dense(self._ntypes, name='output', activation='relu')(hidden)

        self._model = Model(inputs=inputs, outputs=outputs)




    """
    Define your new model here.

    """

    def _fcSmax1d_model(self, patch_size):
        """
        Fully-connected 1d ANN model with soft matrix output layer
        """
        inputs = Input(shape=(self._ndwi,))
        # Define hidden layer
        hidden = Dense(self._kernel1, activation='relu')(inputs)
        for i in np.arange(self._layer  - 1):
            hidden = Dense(self._kernel1, activation='relu')(hidden)

        hidden = Dropout(0.1)(hidden)

        # Define output layer
        outputs = Dense(self._ntypes, name='output', activation='softmax')(hidden)

        self._model = Model(inputs=inputs, outputs=outputs)




    
    def _conv3d_model(self, patch_size):
        """
        Conv3D model.
        """
        if self._train:
            inputs = Input(shape=(patch_size, patch_size, patch_size, self._ndwi))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, dim2, self._ndwi))
        hidden = Conv3D(self._kernel1, 3, activation='relu', padding='valid')(inputs)
        for i in np.arange(self._layer - 1):
            hidden = Conv3D(self._kernel1, 1, activation='relu', padding='valid')(hidden)
        hidden = Dropout(0.1)(hidden)
        outputs = Conv3D(self._ntypes, 1, activation='relu', padding='valid')(hidden)

        self._model = Model(inputs=inputs, outputs=outputs)

    __model = {
        'fc1d' : _fc1d_model,
        'fcSmax1d': _fcSmax1d_model,
        'conv3d' : _conv3d_model,
    }

    def model(self, optimizer, loss, patch_size):
        """
        Generate model.
        """
        self.__model[self._type](self, patch_size)
        self._model.summary()
        self._model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def _sequence_train(self, data, label, nbatch, epochs, callbacks, shuffle, validation_data):

        validation_split = 0.0
        if validation_data is None:
            validation_split = 0.2

        self._hist = self._model.fit(data, label,
                                     batch_size=nbatch,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     validation_data=validation_data,
                                     validation_split=validation_split,
                                     callbacks=callbacks)
        self._loss.append(len(self._hist.history['loss']))
        self._loss.append(self._hist.history['loss'][-1])
        self._loss.append(None)
        self._loss.append(self._hist.history['accuracy'][-1])
        self._loss.append(None)

    __train = {
        'fc1d' : _sequence_train,
        'fcSmax1d': _sequence_train,
        'conv3d' : _sequence_train,
    }

    def train(self, data, label, nbatch, epochs, callbacks, weightname, out_path,
              shuffle=True, validation_data=None):
        """
        Training on training datasets.
        """
        #print "Training start ..."
        self.__train[self._type](self, data, label, nbatch, epochs,
                                 callbacks, shuffle, validation_data)

        # define the output name and path
        if out_path is not None:
            weights_path = os.path.join(out_path, 'weights')
            this_dir = os.getcwdb()
            #out_train = os.path.join(weights_path,  weightname + '.weights')
        else:
            weights_path = os.path.join('weights')
            this_dir = '..'
            #out_train = os.path.join(weights_path, weightname + '.weights')

        if not os.path.isdir(weights_path):
            #os.system("mkdir " + weights_path)
            os.makedirs(weights_path)

        try:

            os.chdir(weights_path)
            # self._model.save_weights(weightname + '.weights')
            self._model.save_weights(weightname)
            os.chdir(this_dir)
        except IOError:
            # If I get error try to save results anyway...
            #os.system('mkdir weights')
            os.makedirs('weights')
            #self._model.save_weights(os.path.join('weights', weightname + '.weights'))
            self._model.save_weights(os.path.join('weights', weightname))

        return self._loss

    def load_weight(self, weightname, out_path):
        """
        Load pre-trained weights.
        """

        # define the output name and path
        if out_path is not None:
            #out_train = os.path.join(out_path, 'weights', weightname + '.weights')
            out_train = os.path.join(out_path, 'weights', weightname)
        else:
            #out_train = os.path.join('weights', weightname + '.weights')
            out_train = os.path.join('weights', weightname)

        self._model.load_weights(out_train)

    def predict(self, data):
        """
        Predict on test datas.
        """
        pred = self._model.predict(data)
        if self._type[-6:] == 'staged':
            pred = np.concatenate((pred[0], pred[1]), axis=-1)

        return pred

def parser():
    """
    Create a parser.
    """
    parser = argparse.ArgumentParser()
    
    # Specify train & test sets
    parser.add_argument("--train_subjects", help="Training subjects IDs", nargs='*')
    parser.add_argument("--test_subject", help="Testing subject ID", nargs='*')
    parser.add_argument("--scheme", metavar='name', help="The scheme for sampling")
    parser.add_argument("--DWI", metavar='N', help="Number of input DWI volumes", type=int, default=60)
    parser.add_argument("--types", metavar='t', help="Type of input labels", nargs='*', default=['NDI' , 'FWF', 'ODI'])
    parser.add_argument("--out", metavar='o', help="Specify out path", type=str, default=None)
  
   # Training parameters
    parser.add_argument("--train", help="Train the network", action="store_true")
    parser.add_argument("--model", help="Train model",
                        choices=['fc1d', 'fcSmax1d', 'conv2d', 'conv3d'], default='fc1d')
    parser.add_argument("--layer", metavar='l', help="Number of layers", type=int, default=3)
    parser.add_argument("--lr", metavar='lr', help="Learning rates", type=float, default=0.001)
    parser.add_argument("--epoch", metavar='ep', help="Number of epoches", type=int, default=100)
    parser.add_argument("--kernels", help="The number of kernels for each layer", nargs='*',
                        type=int, default=None)
    parser.add_argument("--rseed", metavar='rs', help="Random seed", type=int, default=None)
    parser.add_argument("--train_shuffle", metavar='sh', help="Shuffle at each epoch", type=bool, default=False)
    parser.add_argument("--data_shuffle", metavar='sh', help="Shuffle data before training/validation", type=bool, default=False)
    parser.add_argument("--patience", metavar='sh', help="patience", type=int, default=1000)

    # Just For test; not use anymore
    parser.add_argument("--loss", help="Set different loss functions", action="store_true")
    parser.add_argument("--test_shape", nargs='*', type=int, default=None)
    parser.add_argument("--batch", metavar='bn', help="Batch size", type=int, default=256)
    parser.add_argument("--patch_size", metavar='ksize', help="Size of the kernels", type=int, default=3)
    parser.add_argument("--base", metavar='base', help="choice of training data", type=int, default=1)    

    return parser
