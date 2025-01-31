"""
Generate and save .mat data for new datasets.

Author: Ting Gong
"""

import os
import shutil
import numpy as np
from scipy.io import loadmat, savemat
from utils.nii_utils import load_nii_image, save_nii_image, mask_nii_data
import matplotlib.pyplot as plt

def gen_dMRI_fc1d_train_datasets(path, subject, ndwi, scheme, labels, dwi_path, mask_path, combine=None, whiten=True):
    """
    Generate fc1d training Datasets.
    """
    if labels is not None:
        ltype = labels
    else:
        ltype = ['NDI' , 'FWF', 'ODI']

    # os.system("mkdir -p datasets/data datasets/label datasets/mask")
    # os.system('cp ' +  path + '/' + subject + '/nodif_brain_mask.nii datasets/mask/mask_' + subject + '.nii')
    # Modified for compatibility with Windows
    #os.system('md ' + os.path.join('datasets', 'data') + ' ' + os.path.join('datasets', 'label') + ' ' + os.path.join('datasets', 'mask'))
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    if not os.path.isdir(os.path.join('datasets', 'data')):
        os.mkdir(os.path.join('datasets', 'data'))
    if not os.path.isdir(os.path.join('datasets', 'label')):
        os.mkdir(os.path.join('datasets', 'label'))
    if not os.path.isdir(os.path.join('datasets', 'mask')):
        os.mkdir(os.path.join('datasets', 'mask'))

    # Copy and load the mask
    if mask_path is not None:
        # Make sure we use correct file extension to copy
        mask_name, mask_extension = os.path.splitext(mask_path)
        if mask_extension == '.nii':
            m_ext = mask_extension
        elif mask_extension == '.gz':
            m_ext = '.nii' + mask_extension
        else:
            raise Exception("Mask extension {} not known", mask_extension)
        shutil.copy(path + '/' + subject + '/' + mask_path, 'datasets/mask/mask_' + subject + m_ext)
    else:
        m_ext = '.nii'
        shutil.copy(path + '/' + subject + '/nodif_brain_mask.nii', 'datasets/mask/mask_' + subject + m_ext)
    mask = load_nii_image('datasets/mask/mask_' + subject + m_ext)
        
    # load diffusion data
    if dwi_path is not None:
        data = load_nii_image(path + '/' + subject + '/' + dwi_path, mask)
    else:
        data = load_nii_image(path + '/' + subject + '/diffusion.nii', mask)
    
    # Select the inputs.
    if combine is not None:
        data = data[..., combine == 1]
    else:
        data = data[..., :ndwi]

    # Whiten the data.
    if whiten:
        data = data / data.mean() - 1.0
    print(data.shape)

    # load labels
    label = np.zeros((data.shape[0] , len(ltype)))
    for i in range(len(ltype)):
        if labels is not None:
            filename = path + '/' + subject + '/' + ltype[i]
        else:
            filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
        temp = load_nii_image(filename,mask)
        label[:, i] = temp.reshape(temp.shape[0])
    #filename = path + '/' + subject + '/' + subject + '_' + ltype[0] + '.nii'
    #temp = load_nii_image(filename,mask)
    #label[:, 0] = temp.reshape(temp.shape[0])

    #filename = path + '/' + subject + '/' + subject + '_' + ltype[1] + '.nii'
    #temp = load_nii_image(filename,mask) * 1000   # scale MD to the value around 1
    #label[:, 1] = temp.reshape(temp.shape[0])
     
    print(label.shape)

    # remove possible NAN values in parameter maps
    for i in range(label.shape[0]):
        if np.isnan(label[i]).any():
            label[i] = 0
            data[i] = 0

    # save datasets
    savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '1d.mat', {'data':data})
    savemat('datasets/label/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '1d.mat', {'label':label})

def gen_2d_patches(data, mask, size, stride):
    """
    generate 2d patches
    """
    patches = []
    for layer in range(mask.shape[2]):
        for x in np.arange(0, mask.shape[0], stride):
            for y in np.arange(0, mask.shape[1], stride):
                xend, yend = np.array([x, y]) + size
                lxend, lyend = np.array([x, y]) + stride
                if mask[x:lxend, y:lyend, layer].sum() > 0:
                    patches.append(data[x:xend, y:yend, layer, :])

    return np.array(patches)

def gen_3d_patches(data, mask, size, stride):
    """
    generate 3d patches
    """
    #print data.shape, mask.shape
    patches = []
    for layer in np.arange(0, mask.shape[2], stride):
        for x in np.arange(0, mask.shape[0], stride):
            for y in np.arange(0, mask.shape[1], stride):
                xend, yend, layerend = np.array([x, y, layer]) + size
                lxend, lyend, llayerend = np.array([x, y, layer]) + stride
                if mask[x:lxend, y:lyend, layer:llayerend].sum() > 0:
                    patches.append(data[x:xend, y:yend, layer: layerend, :])
    #print np.array(patches).shape
    return np.array(patches)

def gen_dMRI_conv2d_train_datasets(subject, ndwi, scheme, patch_size, label_size, base=1, test=False):
    """
    Generate Conv2D Dataset.
    """
    offset = base - (patch_size - label_size) / 2

    label = loadmat('datasets/label/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat')['label']
    label = label[base:-base, base:-base, base:-base, :]
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
    mask = mask[base:-base, base:-base, base:-base]
    
    data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat')['data']
    data = data[:, :, base:-base, :]

    if offset:
        data = data[offset:-offset, offset:-offset, :, :12]

    patches = gen_2d_patches(data, mask, patch_size, label_size)
    label = gen_2d_patches(label, mask, label_size, label_size)

    savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '2d.mat', {'data':patches})
    savemat('datasets/label/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '2d.mat', {'label':label})

def gen_dMRI_conv3d_train_datasets(subject, ndwi, scheme, patch_size, label_size, base=1, test=False):
    """
    Generate Conv3D Dataset.
    """
    offset = base - (patch_size - label_size) / 2

    label = loadmat('datasets/label/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat')['label']
    label = label[base:-base, base:-base, base:-base, :]
    mask_name = 'datasets/mask/mask_' + subject + '.nii'
    if not os.path.isfile(mask_name):
        mask_name = mask_name + '.gz'
    mask = load_nii_image(mask_name)
    mask = mask[base:-base, base:-base, base:-base]
    
    data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat')['data']

    if offset:
        data = data[offset:-offset, offset:-offset, :, :12]

    patches = gen_3d_patches(data, mask, patch_size, label_size)
    patches = patches.reshape(patches.shape[0], -1)

    label = gen_3d_patches(label, mask, label_size, label_size)

    savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '3d.mat', {'data':patches})
    savemat('datasets/label/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + '3d.mat', {'label':label})

def gen_dMRI_test_datasets(path, subject, ndwi, scheme, labels, dwi_path, mask_path, combine=None,  fdata=True, flabel=True, whiten=True):
    """
    Generate testing Datasets.
    """
    if labels is not None:
        ltype = labels
    else:
        ltype = ['NDI' , 'FWF', 'ODI']

    # os.system("mkdir -p datasets/data datasets/label datasets/mask")
    # os.system('copy ' +  path + '/' + subject + '/nodif_brain_mask.nii datasets/mask/mask_' + subject + '.nii')
    # Modified for compatibility with Windows
    #os.system('md ' + os.path.join('datasets', 'data') + ' ' + os.path.join('datasets', 'label') + ' ' + os.path.join('datasets', 'mask'))
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    if not os.path.isdir(os.path.join('datasets', 'data')):
        os.mkdir(os.path.join('datasets', 'data'))
    if not os.path.isdir(os.path.join('datasets', 'label')):
        os.mkdir(os.path.join('datasets', 'label'))
    if not os.path.isdir(os.path.join('datasets', 'mask')):
        os.mkdir(os.path.join('datasets', 'mask'))

        # Copy and load the mask
    if mask_path is not None:
        # Make sure we use correct file extension to copy
        mask_name, mask_extension = os.path.splitext(mask_path)
        if mask_extension == '.nii':
            m_ext = mask_extension
        elif mask_extension == '.gz':
            m_ext = '.nii' + mask_extension
        else:
            raise Exception("Mask extension {} not known", mask_extension)
        shutil.copy(path + '/' + subject + '/' + mask_path, 'datasets/mask/mask_' + subject + m_ext)
    else:
        m_ext = '.nii'
        shutil.copy(path + '/' + subject + '/nodif_brain_mask.nii', 'datasets/mask/mask_' + subject + m_ext)
    mask = load_nii_image('datasets/mask/mask_' + subject + m_ext)
            
    if fdata:
        # load diffusion data
        if dwi_path is not None:
            data = load_nii_image(path + '/' + subject + '/' + dwi_path)
        else:
            data = load_nii_image(path + '/' + subject + '/diffusion.nii')
        
        # Select the inputs.
        if combine is not None:
            data = data[..., combine == 1]
        else:
            data = data[..., :ndwi]

        # Whiten the data.
        if whiten:
            data = data / data[mask > 0].mean() - 1.0
        
        print(data.shape)
        savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat', {'data':data})

    if flabel:
        label = np.zeros(mask.shape + (len(ltype),))
        for i in range(len(ltype)):
            if labels is not None:
                filename = path + '/' + subject + '/' + ltype[i]
            else:
                filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
            label[:, :, :, i] = load_nii_image(filename)
        #filename = path + '/' + subject + '/' + subject + '_' + ltype[0] + '.nii'
        #label[:, :, :, 0] = load_nii_image(filename)
        #filename = path + '/' + subject + '/' + subject + '_' + ltype[1] + '.nii'
        #label[:, :, :, 1] = load_nii_image(filename) * 1000   # scale MD to the value around 1
        print(label.shape)
        savemat('datasets/label/' + subject+ '-' + str(ndwi) + '-' + scheme + '.mat', {'label':label})

def fetch_train_data_MultiSubject(subjects, model, ndwi, scheme, data_shuffle):
    """
    #Fetch train data.
    """
    data_s = None
    labels = None

    if model[:4] == 'fc1d':
        dim='1d.mat'
    if model[:8] == 'fcSmax1d':
        dim = '1d.mat'
    if model[:6] == 'conv2d':
        dim='2d.mat'
    if model[:6] == 'conv3d':
        dim='3d.mat'

    for subject in subjects:
        label = loadmat('datasets/label/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + dim)['label']
        data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + dim)['data']

        if data_s is None:
            data_s = data
            labels = label
        else:
            data_s = np.concatenate((data_s, data), axis=0)
            labels = np.concatenate((labels, label), axis=0)

    data = np.array(data_s)
    label = np.array(labels)

    # check data distribution
    # fig1, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
    # ax0.hist(label[:1043864, 0], bins=200)
    # ax1.hist(label[:1043864, 1], bins=200)
    # ax2.hist(label[:1043864, 2], bins=200)

    # check if we need to shuffle the data
    if data_shuffle:
        data, label = shuffle_data(data, label.reshape(label.shape[0], label.shape[-1]))
        if model[:6] == 'conv3d':
            label = label.reshape(label.shape[0], 1, 1, 1, -1)


    if model[:6] == 'conv3d':
        data = data.reshape(data.shape[0], 3, 3, 3, -1)

    return data, label

def shuffle_data(data, label):
    """
    Shuffle data.
    """
    size = data.shape[-1]
    datatmp = np.concatenate((data, label), axis=-1)
    np.random.shuffle(datatmp)
    return datatmp[..., :size], datatmp[..., size:]

def repack_pred_label(pred, mask, model, ntype):
    """
    Get.
    """
    if model[7:13] == 'single':
        label = np.zeros(mask.shape + (1,))
    else:
        label = np.zeros(mask.shape + (ntype,))
    
    if model[:6] == 'conv2d':
        label[1:-1, 1:-1, :, :] = pred.transpose(1, 2, 0, 3)
    elif model[:6] == 'conv3d':
        label[1:-1, 1:-1, 1:-1, :] = pred
    else:
        label = pred.reshape(label.shape)
    
    #label[:,:,:,1]=label[:,:,:,1]/1000 # scale MD back while saving nii
    return label