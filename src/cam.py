import tensorflow.keras.backend as K
import tensorflow.keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, load_model
tensorflow.compat.v1.disable_eager_execution()
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import to_time_series_dataset, split_df, load_test, TimeSeriesResampler, TimeSeriesScalerMeanVariance
from scipy.interpolate import interp1d

import seaborn as sns
sns.set(style='white',font='Palatino Linotype',font_scale=1,rc={'axes.grid' : False})


def get_model(id):
    model = load_model('.\\models\\cam_cnn_'+id+'.h5')
    return model


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_data(dataset):
    if dataset == 'test':
        X, y = load_test()
        sz = 230
    elif dataset == 'uc1':
        X, y = split_df(pd.read_pickle('..\\data\\df_uc1.pkl'),
                        index_column='run_id',
                        feature_columns=['fldPosition', 'fldCurrent'],
                        target_name='target')
        # Length of timeseries for resampler and cnn
        sz = 38
    elif dataset == 'uc2':
        X, y = split_df(pd.read_pickle('..\\data\\df_uc2.pkl'),
                        index_column='run_id',
                        feature_columns=['position', 'force'],
                        target_name='label')
        # Length of timeseries for resampler and cnn
        sz = 200
    resampler = TimeSeriesResampler(sz=sz)
    X = resampler.fit_transform(X, y)
    y = np.array(y)
    return X, y


def get_sample(X, y, label, rs=100):
    s = np.random.RandomState(rs)
    s = s.choice(np.where(y == label)[0], 1)
    x_raw = to_time_series_dataset(X[s, :, :])
    scaler = TimeSeriesScalerMeanVariance(kind='constant')
    X = scaler.fit_transform(X)
    x_proc = to_time_series_dataset(X[s, :, :])
    return x_proc, x_raw


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


def grad_cam(input_model, data, category_index, nb_classes, layer_name):
    # Lambda function for getting target category loss
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    # Lambda layer for function
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    # Add Lambda layer as output to model
    model = Model(inputs=input_model.input, outputs=x)
    #model.summary()
    # Function for getting target category loss y^c
    loss = K.sum(model.output)
    # Get the layer with "layer_name" as name
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    # Define function to calculate gradients
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    # Calculate convolution layer output and gradients for datasample
    output, grads_val = gradient_function([data])
    output, grads_val = output[0, :], grads_val[0, :, :]

    # Calculate the neuron importance weights as mean of gradients
    weights = np.mean(grads_val, axis = 0)
    # Calculate CAM by multiplying weights with the respective output
    cam = np.zeros(output.shape[0:1], dtype = np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, i]
    # Interpolate CAM to get it back to the original data resolution
    f = interp1d(np.linspace(0, 1, cam.shape[0]), cam, kind="slinear")
    cam = f(np.linspace(0,1,data.shape[1]))
    # Apply ReLU function to only get positive values
    cam[cam < 0] = 0

    return cam


def plot_grad_cam(cam, raw_input, cmap, alpha, language='eng'):
    fig, ax = plt.subplots(raw_input.shape[-1], 1, figsize=(15, 9), sharex=True)
    # fig.suptitle('Gradient Class Activation Map for sample of class %d' %predicted_class)
    if language == 'eng':
        ax_ylabel = [r"Position $\mathit{z}$ in mm", r"Velocity $\mathit{v}$ in m/s", r"Current $\mathit{I}$ in A"]
    if language == 'ger':
        ax_ylabel = [r"Position $\mathit{z}$ in mm", r"Geschwindigkeit $\mathit{v}$ in m/s", r"Stromstärke $\mathit{I}$ in A"]
    for i, a in enumerate(ax):
        left, right = (-1, raw_input.shape[1] + 1)
        range_input = raw_input[:, :, i].max() - raw_input[:, :, i].min()
        down, up = (raw_input[:, :, i].min() - 0.1 * range_input, raw_input[:, :, i].max() + 0.1 * range_input)
        a.set_xlim(left, right)
        a.set_ylim(down, up)
        a.set_ylabel(ax_ylabel[i])
        im = a.imshow(cam.reshape(1, -1), extent=[left, right, down, up], aspect='auto', alpha=alpha, cmap=cmap)
        a.plot(raw_input[0, :, i], linewidth=2, color='k')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    if language == 'eng':
        cbar_ax.set_ylabel('Activation', rotation=90, labelpad=15)
    if language == 'ger':
        cbar_ax.set_ylabel('Aktivierung', rotation=90, labelpad=15)
    return ax

if __name__ == "__main__":

    X, y = load_data('test')
    nb_classes = np.unique(y).shape[0]
    # Load model and datasample
    preprocessed_input, raw_input = get_sample(X, y, label=1)
    model = get_model('test')

    # Get prediction
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)
    print('Predicted class: ', predicted_class)

    # Calculate Class Activation Map
    cam = grad_cam(model, preprocessed_input, predicted_class, nb_classes, 'block2_conv1')
    ax = plot_grad_cam(cam, raw_input, 'jet', 1)
    plt.show()

