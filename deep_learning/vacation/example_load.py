import sys

sys.path.insert(1, '../../')
from Utils.utils import FileClass, ResultsClass
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras import layers, optimizers
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
import keras.backend as K
from keras.models import Model
import numpy as np


def custom_loss(out):
    def loss(y_true, y_pred):
        weights1 = model.layers[len(model.layers)-1].kernel
        weights2 = model.layers[len(model.layers)-2].kernel
        weights3 = model.layers[len(model.layers)-3].kernel
        mean = tf.math.add(weights1,weights2)
        mean = tf.math.add(mean,weights3)
        mean = tf.multiply(mean,1/3)
        mean = tf.multiply(mean,lamb)
        if out == 0:
            dist = tf.norm(weights1-mean,ord='euclidean')
        if out == 1:
            dist = tf.norm(weights2-mean,ord='euclidean')
        if out == 2:
            dist = tf.norm(weights3-mean,ord='euclidean')
        dist = tf.multiply(dist,dist)
        #dist12 = tf.norm(weights1-weights2, ord='euclidean')
        #dist13 = tf.norm(weights1-weights3, ord='euclidean')
        #dist23 = tf.norm(weights2-weights3, ord='euclidean')
        #dist = tf.math.add(dist12, dist13)
        #dist = tf.math.add(dist, dist23)
        #dist = tf.multiply(tf.multiply(dist,dist) , .lamb)
        loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
        res = tf.math.add(loss , dist)
        mask = tf.reduce_all(tf.equal(y_true[0][0], out))



culture = 0
epochs = 1
verbose_param = 0
learning_rate = 0.00001
lamb = 1
validation_split = 0.2
batch_size = 1

size = (100,100,3)
input = Input(size)
x = tf.keras.Sequential([
    ResNet50(input_shape=size, weights='imagenet', include_top=False)
])(input)
x = Flatten()(x)
y1 = (Dense(1, activation='sigmoid', name='dense'))(x)
y2 = (Dense(1, activation='sigmoid', name='dense_1'))(x)
y3 = (Dense(1, activation='sigmoid', name='dense_2'))(x)
model = Model(inputs=input,
            outputs = [y1,y2,y3],
            name = 'model')
model.trainable = True
for layer in model.layers[1].layers:
    layer.trainable = False
for layer in model.layers[1].layers[-3:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True
if culture == 0:
    monitor_val = 'val_dense_accuracy'
else:
    monitor_val = f'val_dense_{culture}_accuracy'
lr_reduce = ReduceLROnPlateau(monitor=monitor_val,
                                factor=0.1,
                                patience=int(epochs/3) + 1,
                                verbose=verbose_param,
                                mode='max',
                                min_lr=1e-8)
early = EarlyStopping(monitor=monitor_val,
                        min_delta=0.001,
                        patience=int(epochs/1.7) + 1,
                        verbose=verbose_param,
                        mode='auto')
adam = optimizers.Adam(learning_rate)
optimizer = adam

model.compile(optimizer=optimizer,
                metrics=["accuracy"],
                loss=[custom_loss(out=0),custom_loss(out=1),custom_loss(out=2)])

model2 = model

# I WANT TO CHECK THAT THE WEIGHTS ARE LOADABLE AND THAT THEY ARE DIFFERENT
checkpoint_path = "./c_ind_mit/0.05/-1/checkpoint_0"
checkpoint_path2 = "./c_ind_mit/0.05/-1/checkpoint_17"
model.load_weights(checkpoint_path)
#model.summary()
model2.load_weights(checkpoint_path2)

X = np.random.rand(10, 100, 100, 3)
y = np.random.rand(10,  1)

y_pred = model(X)
y_pred2 = model2(X)

for y_i in y_pred:

    print(confusion_matrix(np.round(y), np.round(y_i)))
for y_i in y_pred2:
    print(confusion_matrix(np.round(y), np.round(y_i)))

def custom_loss(out):
    def loss(y_true, y_pred):
        weights1 = model.layers[len(model.layers)-1].kernel
        weights2 = model.layers[len(model.layers)-2].kernel
        weights3 = model.layers[len(model.layers)-3].kernel
        mean = tf.math.add(weights1,weights2)
        mean = tf.math.add(mean,weights3)
        mean = tf.multiply(mean,1/3)
        mean = tf.multiply(mean,lamb)
        if out == 0:
            dist = tf.norm(weights1-mean,ord='euclidean')
        if out == 1:
            dist = tf.norm(weights2-mean,ord='euclidean')
        if out == 2:
            dist = tf.norm(weights3-mean,ord='euclidean')
        dist = tf.multiply(dist,dist)
        #dist12 = tf.norm(weights1-weights2, ord='euclidean')
        #dist13 = tf.norm(weights1-weights3, ord='euclidean')
        #dist23 = tf.norm(weights2-weights3, ord='euclidean')
        #dist = tf.math.add(dist12, dist13)
        #dist = tf.math.add(dist, dist23)
        #dist = tf.multiply(tf.multiply(dist,dist) , .lamb)
        loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
        res = tf.math.add(loss , dist)
        mask = tf.reduce_all(tf.equal(y_true[0][0], out))

def get_weights_print_stats(layer):
    W = layer.get_weights()
    print(len(W))
    for w in W:
        print(w.shape)
    return W

def hist_weights(weights, bins=500):
    for weight in weights:
        plt.hist(np.ndarray.flatten(weight), bins=bins)
        #plt.show()

W = get_weights_print_stats(model.layers[-1])
print(W)

hist_weights(W)

W2 = get_weights_print_stats(model2.layers[-1])

hist_weights(W2)

print(np.array_equal(W,W2)) #False