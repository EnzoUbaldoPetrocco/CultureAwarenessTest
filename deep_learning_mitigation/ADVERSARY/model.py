import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.layers import Dense, Flatten, Input
from tensorflow.keras import Model
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import numpy as np
from matplotlib import pyplot as plt
import time

# MODEL
class Net(Model):
    def __init__(self, size):
        super(Net, self).__init__()

        self.inp = Input(size)
        self.base = tf.keras.Sequential([
            ResNet50(input_shape=size, weights='imagenet', include_top=False)
        ])
        self.flatten = Flatten()
        self.y1 = (Dense(1, activation='sigmoid', name='dense'))
        self.y2 = (Dense(1, activation='sigmoid', name='dense_1'))
        self.y3 = (Dense(1, activation='sigmoid', name='dense_2'))

    def call(self, x):
        x = self.inp(x)
        x = self.base(x)
        x = self.base(x)
        x = self.flatten(x)
        return [self.y1(x), self.y2(x), self.y3(x)]

class MitigationModel:
    def __init__(self, lr, lamb, bs, nb_epochs, eps):
        self.model = Net()
        self.loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)
        # Metrics to track the different accuracies.
        self.train_loss = tf.metrics.Mean(name="train_loss")
        self.test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
        self.test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
        self.test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()
        # Hyperparams
        self.lr = lr #learning rate
        self.lamb = lamb #gain of regularizer
        self.batch_size = bs #batch size
        self.nb_epochs= nb_epochs #number of epochs
        self.eps = eps #step of adv

    @tf.function
    def train_step(self,x, y_true):
        def loss(y_true, y_pred, out):
            weights1 = self.model.layers[len(self.model.layers) - 1].kernel
            weights2 = self.model.layers[len(self.model.layers) - 2].kernel
            weights3 = self.model.layers[len(self.model.layers) - 3].kernel
            mean = tf.math.add(weights1, weights2)
            mean = tf.math.add(mean, weights3)
            mean = tf.multiply(mean, 1 / 3)
            mean = tf.multiply(mean, self.lamb)
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
            y_pred = self.model(x)
            for culture, y_hat in enumerate(y_pred):
                loss = loss(y_true, y_hat, culture)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)

    def fit(self, data, adv_train):
        # Train model with adversarial training
        for epoch in range(self.nb_epochs):
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(60000)
            for (x, y) in data.train:
                if adv_train:
                    # Replace clean example with adversarial example for adversarial training
                    x = projected_gradient_descent(self.model, x, self.eps, 0.01, 40, np.inf)
                self.train_step(x, y)
                progress_bar_train.add(x.shape[0],
                                    values=[("loss", self.train_loss.result())])

    def test(self, data):
        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(10000)
        size = np.shape(data.test[0])
        for culture in range(3):
            for x, y in data.test[culture]:
                y_pred = self.model(x)
                self.test_acc_clean(y, y_pred)

                x_fgm = fast_gradient_method(self.model, x, self.eps, np.inf)
                y_pred_fgm = self.model(x_fgm)
                self.test_acc_fgsm(y, y_pred_fgm)

                x_pgd = projected_gradient_descent(self.model, x, self.eps, 0.01, 40, np.inf)
                y_pred_pgd = self.model(x_pgd)
                self.test_acc_pgd(y, y_pred_pgd)

                progress_bar_test.add(x.shape[0])

            print(f"CULTURE {culture}" + " test acc on clean examples           (%): {:.3f}".format(
                self.test_acc_clean.result() * 100))
            print(f"CULTURE {culture}" + " test acc on FGM adversarial examples (%): {:.3f}".format(
                self.test_acc_fgsm.result() * 100))
            print(f"CULTURE {culture}" + " test acc on PGD adversarial examples (%): {:.3f}".format(
                self.test_acc_pgd.result() * 100))

            for x, y in data.test:
                for i in range(x.shape[0]):
                    y_tmp = y[i]
                    x_tmp = tf.reshape(x[i, :, :, :], [1, size[0], size[1], size[2]])
                    y_pred = self.model(x_tmp)
                    x_fgm = fast_gradient_method(self.model, x_tmp, self.eps, np.inf)
                    y_pred_fgm = self.model(x_fgm)
                    x_pgd = projected_gradient_descent(self.model, x_tmp, self.eps, 0.01, 40,
                                                    np.inf)
                    y_pred_pgd = self.model(x_pgd)
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