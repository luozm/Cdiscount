# Convert SE-ResNet-50 from Caffe to Keras
# Using the model from https://github.com/shicai/SENet-Caffe

import os
import numpy as np

# The caffe module needs to be on the Python path; we'll add it here explicitly.
import sys
caffe_root = "/path/to/caffe"
sys.path.insert(0, caffe_root + "python")
import caffe

model_root = "/path/to/SE-ResNet-50/"
model_def = model_root + 'se_resnet_50_v1_deploy.prototxt'
model_weights = model_root + 'se_resnet_50_v1.caffemodel'

if not os.path.isfile(model_weights):
    print("Model not found")
    
caffe.set_mode_cpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)

fc_layers = [
    "fc2_1/sqz", "fc2_1/exc",
    "fc2_2/sqz", "fc2_2/exc",
    "fc2_3/sqz", "fc2_3/exc",

    "fc3_1/sqz", "fc3_1/exc",
    "fc3_2/sqz", "fc3_2/exc",
    "fc3_3/sqz", "fc3_3/exc",
    "fc3_4/sqz", "fc3_4/exc",

    "fc4_1/sqz", "fc4_1/exc",
    "fc4_2/sqz", "fc4_2/exc",
    "fc4_3/sqz", "fc4_3/exc",
    "fc4_4/sqz", "fc4_4/exc",
    "fc4_5/sqz", "fc4_5/exc",
    "fc4_6/sqz", "fc4_6/exc",

    "fc5_1/sqz", "fc5_1/exc",
    "fc5_2/sqz", "fc5_2/exc",
    "fc5_3/sqz", "fc5_3/exc",

    "fc6",
]

real_name = None
mean = None
variance = None
bias = None
params = {}

for layer_name, param in net.params.items():
    shapes = map(lambda x: x.data.shape, param)
    print(layer_name.ljust(25) + str(list(shapes)))
    
    # Dense layer with bias.
    if layer_name in fc_layers:
        # Caffe stores the weights as (outputChannels, inputChannels).
        c_o  = param[0].data.shape[0]
        c_i  = param[0].data.shape[1]        

        # Keras on TensorFlow uses: (inputChannels, outputChannels).
        weights = np.array(param[0].data.data, dtype=np.float32).reshape(c_o, c_i)
        weights = weights.transpose(1, 0)

        bias = param[1].data
        params[layer_name] = [weights, bias]
   
    # These are the batch norm parameters.
    # Each BatchNorm layer has three blobs:
    #   0: mean
    #   1: variance
    #   2: scale factor
    elif "/bn" in layer_name:
        factor = param[2].data[0]
        mean = np.array(param[0].data, dtype=np.float32) / factor
        variance = np.array(param[1].data, dtype=np.float32) / factor
        real_name = layer_name

    # This is a scale layer. It always follows BatchNorm.
    # A scale layer has two blobs:
    #   0: scale (gamma)
    #   1: bias (beta)
    elif "/scale" in layer_name:
        gamma = np.array(param[0].data, dtype=np.float32)
        beta = np.array(param[1].data, dtype=np.float32)

        if real_name is None: print("*** ERROR! ***")
        if mean is None: print("*** ERROR! ***")
        if variance is None: print("*** ERROR! ***")

        params[real_name] = [gamma, beta, mean, variance]

        real_name = None
        mean = None
        variance = None
        bias = None

    # Conv layer with batchnorm, no bias
    else:
        # The Caffe model stores the weights for each layer in this shape:
        # (outputChannels, inputChannels, kernelHeight, kernelWidth)
        c_o  = param[0].data.shape[0]
        c_i  = param[0].data.shape[1]
        h    = param[0].data.shape[2]
        w    = param[0].data.shape[3]

        # Keras on TensorFlow expects weights in the following shape:
        # (kernelHeight, kernelWidth, inputChannels, outputChannels)
        weights = np.array(param[0].data.data, dtype=np.float32).reshape(c_o, c_i, h, w)
        weights = weights.transpose(2, 3, 1, 0)
        params[layer_name] = [weights]
        
np.save("SENet_params.npy", params)
