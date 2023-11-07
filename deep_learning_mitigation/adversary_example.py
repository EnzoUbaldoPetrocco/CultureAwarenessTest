# pip install cleverhans
# pip install easydict
# LIBRARIES
import sys

sys.path.insert(1, '../')
import numpy as np
import tensorflow as tf
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras import Model
from keras.applications.resnet import ResNet50
from keras.layers import Dense, Flatten, Input
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import DS.ds
import time
import matplotlib.pyplot as plt
from deep_learning_mitigation.strings import Strings



# MODEL
class Net(Model):
    def __init__(self, size):
        super(Net, self).__init__()

        self.inp = Input(size)
        self.base = tf.keras.Sequential([
            ResNet50(input_shape=size, weights='imagenet', include_top=False)
        ])
        self.flatten = Flatten()(x)
        self.y1 = (Dense(1, activation='sigmoid', name='dense'))
        self.y2 = (Dense(1, activation='sigmoid', name='dense_1'))
        self.y3 = (Dense(1, activation='sigmoid', name='dense_2'))

    def call(self, x):
        x = self.inp(x)
        x = self.base(x)
        x = self.base(x)
        x = self.flatten(x)
        return [self.y1(x), self.y2(x), self.y3(x)]


# LOAD DS
def load_ds(paths, greyscale, culture, percent, batch):
    def split_list(lst, chunk_size):

        return list(zip(*[iter(lst)] * chunk_size))

    obj = DS.ds.DSClass()
    obj.mitigation_dataset(paths, greyscale, 0)
    obj.nineonedivision(culture, percent=percent)
    # I have to select a culture
    TS = obj.TS[culture]
    # I have to test on every culture
    TestSets = obj.TestS
    TS = split_list(TS, batch)
    print(np.shape(TestSets))
    for k, TestSet in enumerate(TestSets):
        #print(np.shape(TestSets))
        #print(np.shape(TestSet))
        TestSets[k] = split_list(TestSet, batch)
    return EasyDict(train=TS, test=TestSets)


nb_epochs = 8  # Number of epochs
eps = 0.3  # Total epsilon for FGM and PGD attacks
adv_train = True  # Use adversarial training (on PGD adversarial examples)



strings = Strings()
paths = strings.carpet_paths_str

greyscale = 0
culture = 0
percent = 0.05
batch = 4
lr = 0.0001
lamb = 1

data = load_ds(paths, greyscale, culture, percent, batch)
size = np.shape(data.train)
print(size)
model = Net(size)
loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=lr)

# Metrics to track the different accuracies.
train_loss = tf.metrics.Mean(name="train_loss")
test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(x, y_true):
    def loss(y_true, y_pred, out):
        weights1 = model.layers[len(model.layers) - 1].kernel
        weights2 = model.layers[len(model.layers) - 2].kernel
        weights3 = model.layers[len(model.layers) - 3].kernel
        mean = tf.math.add(weights1, weights2)
        mean = tf.math.add(mean, weights3)
        mean = tf.multiply(mean, 1 / 3)
        mean = tf.multiply(mean, lamb)
        if out == 0:
            dist = tf.norm(weights1 - mean, ord='euclidean')
        if out == 1:
            dist = tf.norm(weights2 - mean, ord='euclidean')
        if out == 2:
            dist = tf.norm(weights3 - mean, ord='euclidean')
        dist = tf.multiply(dist, dist)
        loss = tf.keras.losses.binary_crossentropy(y_true[0][1], y_pred[0])
        res = tf.math.add(loss, dist)
        mask = tf.reduce_all(tf.equal(y_true[0][0], out))

        if not mask:
            return 0.0
        else:
            return res

    with tf.GradientTape() as tape:
        # TODO:
        # apply custom loss
        y_pred = model(x)
        loss = loss(y_true, y_pred, culture)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


# Train model with adversarial training
for epoch in range(nb_epochs):
    # keras like display of progress
    progress_bar_train = tf.keras.utils.Progbar(60000)
    for (x, y) in data.train:
        if adv_train:
            # Replace clean example with adversarial example for adversarial training
            x = projected_gradient_descent(model, x, eps, 0.01, 40, np.inf)
        train_step(x, y)
        progress_bar_train.add(x.shape[0],
                               values=[("loss", train_loss.result())])

# Evaluate on clean and adversarial data
progress_bar_test = tf.keras.utils.Progbar(10000)
for culture in range(3):
    for x, y in data.test[culture]:
        y_pred = model(x)
        test_acc_clean(y, y_pred)

        x_fgm = fast_gradient_method(model, x, eps, np.inf)
        y_pred_fgm = model(x_fgm)
        test_acc_fgsm(y, y_pred_fgm)

        x_pgd = projected_gradient_descent(model, x, eps, 0.01, 40, np.inf)
        y_pred_pgd = model(x_pgd)
        test_acc_pgd(y, y_pred_pgd)

        progress_bar_test.add(x.shape[0])

    print(f"CULTURE {culture}" + " test acc on clean examples           (%): {:.3f}".format(
        test_acc_clean.result() * 100))
    print(f"CULTURE {culture}" + " test acc on FGM adversarial examples (%): {:.3f}".format(
        test_acc_fgsm.result() * 100))
    print(f"CULTURE {culture}" + " test acc on PGD adversarial examples (%): {:.3f}".format(
        test_acc_pgd.result() * 100))

    for x, y in data.test:
        for i in range(x.shape[0]):
            y_tmp = y[i]
            x_tmp = tf.reshape(x[i, :, :, :], [1, size[0], size[1], size[2]])
            y_pred = model(x_tmp)
            x_fgm = fast_gradient_method(model, x_tmp, eps, np.inf)
            y_pred_fgm = model(x_fgm)
            x_pgd = projected_gradient_descent(model, x_tmp, eps, 0.01, 40,
                                               np.inf)
            y_pred_pgd = model(x_pgd)
            print("Label on original imput:                            %d" %
                  y_tmp.numpy())
            print("Label on fast gradient method modified imput:       %d" %
                  np.argmax(y_pred_fgm))
            print("Label on projected gradient descent modified imput: %d" %
                  np.argmax(y_pred_pgd))
            fig = plt.figure(figsize=(10, 40))
            fig.add_subplot(1, 3, 1), plt.imshow(x_tmp.numpy().reshape(
                (size[0], size[1])),
                                                 cmap=plt.cm.gray,
                                                 origin='upper',
                                                 interpolation='none')
            fig.add_subplot(1, 3, 2), plt.imshow(x_fgm.numpy().reshape(
                (size[0], size[1])),
                                                 cmap=plt.cm.gray,
                                                 origin='upper',
                                                 interpolation='none')
            fig.add_subplot(1, 3, 3), plt.imshow(x_fgm.numpy().reshape(
                (size[0], size[1])),
                                                 cmap=plt.cm.gray,
                                                 origin='upper',
                                                 interpolation='none')
            plt.show()
            time.sleep(10)