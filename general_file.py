from pathlib import Path
import pathlib
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import cv2
import math
import csv
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from tf_explain.utils.display import grid_display, heatmap_display
import gc
from keras.models import Model
import random
import time
from datetime import datetime
from keras import layers
from keras.regularizers import Regularizer
random.seed(int(datetime.now().timestamp()))
tf.random.set_seed(int(datetime.now().timestamp()))

class FileManagerClass:
    def __init__(self, name, create=True):
        self.name = name
        dir = os.path.dirname(name)
        if create:
            self.mkdir(dir)
    def mkdir(self, dir):
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except Exception as e:
            pass 
    def readrows(self):
        csvlist = []
        try:
            with open(self.name, "r") as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    csvlist.append(row)                
                file.close()
        except:
            pass
        return csvlist
    def writerow(self, row, discriminator=0):
        try:
            with open(self.name, "a", newline="") as file:
                if discriminator:
                    row.to_csv(file)
                    file.write('\n')
                else:
                    writer = csv.writer(file)
                    writer.writerow(row)
                file.close()
        except Exception as e:
            pass 
    def writecm(self, cm, discriminator=0):
        if discriminator:
            class_labels = ['Class 0', 'Class 1', 'Class 2']
            row = pd.DataFrame(cm, columns=class_labels, index=class_labels)
            row['Matrix'] = 'CM'
            row = row[['Matrix'] + class_labels]
        else:
            row = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        self.writerow(row, discriminator=discriminator)
    def readcms(self):
        cms = []
        rows = self.readrows()
        for row in rows:
            cm = [[int(row[0]), int(row[1])], [int(row[2]), int(row[3])]]
            cms.append(cm)
        return cms

class DataClass:

    def __init__(self, paths) -> None:
        self.dataset = []
        for j, path in enumerate(paths):
            labels = self.get_labels(path)
            imgs_per_culture = []
            for i, label in enumerate(labels):
                images = self.get_images(path + "/" + label)
                X = []
                for k in range(len(images)):
                    X.append([images[k], [j, i]])
                imgs_per_culture.append(X)
            self.dataset.append(imgs_per_culture)

    def get_labels(self, path):
      
        dir_list = []
        for file in os.listdir(path):
            d = os.path.join(path, file)
            if os.path.isdir(d):
                d = d.split("\\")
                if len(d) == 1:
                    d = d[0].split("/")
                d = d[-1]
                dir_list.append(d)
        return dir_list

    def get_images(self, path, n=1000, rescale=False):
       
        images = []
        types = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for typ in types:
            paths.extend(pathlib.Path(path).glob(typ))
        paths = paths[0 : min(len(paths), n)]
        for i in paths:
            im = cv2.imread(str(i)) 
            if rescale:
                im = im  /255
            im = im[..., ::-1]
            images.append(im)
        return images

    
    def prepare(
        self,
        culture,
        percent=0,
        shallow=0,
        val_split=0.2,
        test_split=0.2,
        n=1000,
        n_cultures=3,
        imbalanced=0
    ):
        
        self.X = []
        self.y = []
        self.Xv = []
        self.yv = []
        self.Xt = []
        self.yt = []
        for c, cDS in enumerate(self.dataset):
            cultureXt = []
            cultureyT = []
            for lb, lDS in enumerate(cDS):
                random.shuffle(lDS)
                Xds = []
                yds = []
                for img, label in lDS:
                    a = np.zeros(n_cultures)
                    a[c] = 1
                    a = np.append(a, label[1])
                    label=list(a) 
                    if shallow:
                        img = img[0::]
                        img = img.flatten()
                    Xds.append(img)
                    yds.append(label)
                nt = int(n * test_split)
                cultureXt.extend(Xds[0:nt])
                cultureyT.extend(yds[0:nt])

                Xds = Xds[nt : n - 1]
                yds = yds[nt : n - 1]
                if percent != 0:
                    if c != culture:
                        Xds = Xds[0 : int(percent * len(Xds))]
                        yds = yds[0 : int(percent * len(yds))]
                    nv = int(val_split * len(Xds))
                    self.Xv.extend(Xds[0:nv])
                    self.yv.extend(yds[0:nv])
                    self.X.extend(Xds[nv : len(Xds)])
                    self.y.extend(yds[nv : len(yds)])
                else:
                    if c == culture:
                        nv = int(val_split * len(Xds))
                        self.Xv.extend(Xds[0:nv])
                        self.yv.extend(yds[0:nv])
                        self.X.extend(Xds[nv : len(Xds)])
                        self.y.extend(yds[nv : len(yds)])
            self.Xt.append(cultureXt)
            self.yt.append(cultureyT)

    def clear(self):
        self.X = None
        self.y = None
        self.Xv = None
        self.yv = None
        self.Xt = None

class PreprocessingClass:
    def classical_augmentation(self, X, g=0.1, n=-1):
        
        if n <= 0 or n == None:
            n = len(X)
        X = X[0:n]
        X = np.asarray(X)

        shape = np.shape(X[0])
        data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.01),
                    layers.GaussianNoise(g),
                    layers.RandomZoom(g, g),
                    layers.Resizing(shape[0], shape[1]),
                ]
            )
        
        X_augmented = data_augmentation(X, training=True)
        return np.asarray(X_augmented)

class DeepStrings:
    def __init__(self, search_root=None):
 
        if not search_root:
            search_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
        found = False
        for root, subdirs, files in os.walk(search_root):
            if found:
                break
            if not found:
                for d in subdirs:
                    if d == "FINALDS":
                        rt = os.path.abspath(os.path.join(root, d))
                        found = True

        self.lamp_paths = [
            rt + "/lamps/chinese/120/RGB",
            rt + "/lamps/french/120/RGB",
            rt + "/lamps/turkish/120/RGB",
        ]

        self.carpet_paths_str = [
            rt + "/carpets_stretched/indian/200/RGB",
            rt + "/carpets_stretched/japanese/200/RGB",
            rt + "/carpets_stretched/scandinavian/200/RGB",
        ]

        self.carpet_paths_bla = [
            rt + "/carpets_blanked/indian/200/RGB",
            rt + "/carpets_blanked/japanese/200/RGB",
            rt + "/carpets_blanked/scandinavian/200/RGB",
        ]

class CustomReg(Regularizer):
    def __init__(self, lamb, n_cultures):
        self.lamb = lamb
        self.n_cultures = n_cultures

    def __call__(self, x):
        
        mean = tf.reshape(tf.reduce_mean(x, axis=1),  [-1, 1])
        diff = tf.subtract(x, mean)
        reg = tf.reduce_sum(tf.square(diff))
        res = (self.lamb) * reg

        return res
        
class GeneralModelClass:
    
    def __init__(self, standard=0, n_cultures=3, adversarial=0, imbalanced=0) -> None:
        
        self.model = Model()
        self.standard = standard
        self.n_cultures = n_cultures
        self.adversarial=adversarial
        self.imbalanced=imbalanced

    def __call__(self, X, out=-1):
        
        if self.model != None:
                res = self.model.predict(np.asarray(X, dtype='int32'))
                if not self.standard:
                    #print("Res before")
                    #print(res)
                    #print(f"Out:")
                    #print(out)
                    res = np.asarray(res, dtype=np.float32)[out][:, 0]
                    #print("Res after")
                    #print(res)
                return res
        else:
            print("Try fitting the model before")
            return None

    def quantize(self, yF):
        
        values = []
        for y in yF:
            if y > 0.5:
                values.append(1)
            else:
                values.append(0)
            gc.collect()
        return values

    def test(self, Xt, out=-1):
        
        if self.model:
            yF = self(Xt, out)
            yFq = self.quantize(yF)
            gc.collect()
            return yFq
        else:
            gc.collect()
            print("Try fitting the model before")
            return None

    def get_model_stats(self, Xt, yT, out=-1, discriminator=0, j=-1):
        
        if discriminator==0:
            yFq = self.test(Xt, out)
            if len(np.shape(yT)) > 1:
                if type(yT) == list:
                    yT = np.asarray(yT)
                if self.standard:
                    if not self.adversarial:
                        yT = yT[:, 1]
                    else:
                        yT = yT[:, self.n_cultures]
                else:
                    yT = yT[:, self.n_cultures]
                gc.collect()
            gc.collect()
            if yFq!=None:
                cm = confusion_matrix(y_true=yT, y_pred=yFq)
                print(f"Confusion Matrix: {cm}")
                return cm
        else:
            X = []
            for XC in Xt:
                X.extend(XC)
            y = []
            for yC in yT:
                y.extend(yC)
            yF = self.model.predict(np.asarray(X, dtype='int32'))
            yF = np.argmax(yF, axis=1)
            y = np.asarray(y)
            y = y[:,0: self.n_cultures]
            y = np.argmax(y, axis=1)
            if yF.any()!=None:
                cm = confusion_matrix(y_true=y, y_pred=yF)
                print(f"Confusion Matrix: {cm}")
                return cm
        
    def get_model_from_weights(self, path="./"):
        self.model = tf.keras.models.load_model(path)

    def save_model(self, path="./"):
        self.model.save(path)
        



    def explain(
        self,
        validation_data,
        class_index,
        layer_name=None,
        use_guided_grads=True,
        colormap=cv2.COLORMAP_VIRIDIS,
        image_weight=0.7,
    ):
        
        images, _ = validation_data

        if layer_name is None:
            layer_name = self.infer_grad_cam_target_layer()

        outputs, grads = self.get_gradients_and_filters(images, layer_name, class_index, use_guided_grads)

        print(f"outputs in explain: {np.shape(outputs)}")
        print(f"grads in explain: {np.shape(grads)}")

        cams = self.generate_ponderated_output(outputs, grads)

        heatmaps = np.array(
            [
                # not showing the actual image if image_weight=0
                heatmap_display(cam.numpy(), image, colormap, image_weight)
                for cam, image in zip(cams, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    #@staticmethod
    def infer_grad_cam_target_layer(self):
        
        for layer in reversed(self.model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )

    #@staticmethod
    def get_gradients_and_filters(
        self, images, layer_name, class_index, use_guided_grads
    ):
        
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.layers[2].get_layer(layer_name).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        if use_guided_grads:
            grads = (
                tf.cast(conv_outputs > 0, "float32")
                * tf.cast(grads > 0, "float32")
                * grads
            )

        return conv_outputs, grads

    #@staticmethod
    def generate_ponderated_output(self, outputs, grads):
        maps = [
            self.ponderate_output(output, grad)
            for output, grad in zip(outputs, grads)
        ]

        return maps

    #@staticmethod
    def ponderate_output(self, output, grad):
        weights = tf.reduce_mean(grad, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        return cam
    
    def save(self, img, outdir, name):
        cv2.imwrite(img, outdir + name + ".jpg")

    def test_gradcam(self,gradcam_layers, Xv, yv, out_dir):
        for name in gradcam_layers:
                    for class_index in range(2):
                        print(f"Shape of Xv is {np.shape(Xv)}")
                        print(f"Shape of yv is {np.shape(yv)}")
                        output = self.explain(validation_data=(Xv, yv),
                                                class_index=class_index,
                                                layer_name=name)
                        # Save output
                        self.save(output, out_dir, name)

class MitigatedModelsAdvanced(GeneralModelClass):
    def __init__(
        self,
        type="DL",
        culture=0,
        verbose_param=0,
        epochs=15,
        batch_size=1,
        learning_rate=1e-3,
        lambda_index=-1,
        n_cultures = 3,
        weights=None,
        imbalanced=0,
        diffusion=0,
        parify_batches_diffusion=0
    ):
        
        GeneralModelClass.__init__(self, n_cultures=n_cultures, imbalanced=imbalanced)
        self.type = type
        self.culture = culture
        self.verbose_param = verbose_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weights=np.ones(self.n_cultures)
        self.diffusion=diffusion
        self.parify_batches_diffusion=parify_batches_diffusion
        if weights is not None:
            self.weights=weights

        if lambda_index >= 0:
            lambda_grid = np.logspace(-3, 2, 31)
            self.lamb = lambda_grid[lambda_index]
        else:
            self.lamb = 0

    def computeCIC(self, errs):
        tf.add(errs, -tf.math.reduce_min(errs))
        cic = tf.reduce_mean(errs)
        return cic

    def get_cic(self, valX, valY):
        losses = []
        valY = list(np.asarray(valY)[:, self.n_cultures])
        for out in range(self.n_cultures):
            yPred = self.model.predict(np.asarray(valX, dtype="int32"))
            yPred = list(np.asarray(yPred)[:, out])
            ls = tf.keras.losses.binary_crossentropy(valY, yPred)
            losses.append(ls)
        cic = float(self.computeCIC(losses))
        return cic

    def parify_batches(self, s, culture, batch_size):
        indeces_per_culture = []
        
        for i in range(self.n_cultures):
            c = np.zeros(self.n_cultures)
            c[i] = 1.0
            
            vals = np.where((np.asarray(s[1], dtype=object)[:, :self.n_cultures] == c).all(axis=1))[0]
            indeces_per_culture.append(vals)
            if i == culture:
                n_samples_majority = len(vals)
            
        B = []
        Blabel = []
        DS = []
        DSlabel = []
        for i in range(n_samples_majority):
            for j in range(self.n_cultures):
                if len(B)>= batch_size:
                    DS.append(np.asarray(B))
                    B = []
                    DSlabel.append(np.asarray(Blabel))
                    Blabel = []
                sample = np.asarray(s[0])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                label = np.asarray(s[1])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                B.append(sample)
                Blabel.append(label)

        if len(B)>0:
            for i in range(batch_size-len(B)):
                j = i % self.n_cultures
                rnd_index = np.random.randint(0, len(indeces_per_culture[j]))
                B.append(s[0][indeces_per_culture[j][rnd_index]])
                Blabel.append(s[1][indeces_per_culture[j][rnd_index]])
            DS.append(np.asarray(B))
            DSlabel.append(np.asarray(Blabel))

        return DS, DSlabel
  
    def parify_batches_fast(self, s, culture, batch_size):

        X = np.asarray(s[0])
        y = np.asarray(s[1])
        culture_labels = y[:, :self.n_cultures]
        indices_per_culture = [
            np.where(culture_labels[:, i] == 1)[0]
            for i in range(self.n_cultures)
        ]
        n_majority = len(indices_per_culture[culture])
        all_indices = []
        for i in range(self.n_cultures):
            idxs = indices_per_culture[i]
            tiled = np.resize(idxs, n_majority)
            all_indices.append(tiled)
        all_indices = np.stack(all_indices, axis=1)
        final_indices = all_indices.reshape(-1)
        np.random.shuffle(final_indices)
        X_balanced = X[final_indices]
        y_balanced = y[final_indices]

        return X_balanced, y_balanced

    def custom_loss(self):
        n_cultures = self.n_cultures
        lamb = self.lamb
        model = self.model

        def reg():
            chinese_weights = model.get_layer('pred_dense_layer_0').kernel
            french_weights = model.get_layer('pred_dense_layer_1').kernel
            turkish_weights = model.get_layer('pred_dense_layer_2').kernel
            all_weights = tf.concat([chinese_weights, french_weights, turkish_weights], axis=0)
            mean_all_weights = tf.reduce_mean(all_weights)
            regularization_term = tf.reduce_sum(tf.square(all_weights - mean_all_weights))
            return regularization_term

        def loss(y_true, y_pred_list):
            y_pred = tf.concat(y_pred_list, axis=1)  
            y_true_label = y_true[:, n_cultures]
            culture_selector = y_true[:, 0:n_cultures]
            selected_y_pred = tf.reduce_sum(culture_selector * y_pred, axis=1)


            bc = tf.keras.losses.binary_crossentropy(y_true_label, selected_y_pred)
            mean_bc = tf.reduce_mean(bc)
            total_loss = mean_bc + lamb * reg()
            return total_loss

        return loss

    def custom_accuracy(self):
        n_cultures = self.n_cultures
        def accuracy(y_true, y_pred):
            y_pred = tf.concat(y_pred, axis=1)  
            y_true_label = y_true[:, n_cultures]
            culture_selector = y_true[:, 0:n_cultures]
            selected_y_pred = tf.reduce_sum(culture_selector * y_pred, axis=1)

            acc = tf.keras.metrics.binary_accuracy(y_true_label, selected_y_pred)
            
            return acc

        return accuracy
    
    def ImbalancedTransformation(self, TS):
        newX = []
        newY = []
        X = TS[0]
        Y = TS[1]
        for i in range(len(X)):
            img = X[i]
            label = Y[i]
            label = label[0:self.n_cultures]
            if np.sum(label)>0:
                label = np.argmax(label)
                for j in range(int(1/self.weights[label])): 
                    newX.append(img) 
                    newY.append(Y[i])
        return (newX, newY)
    def regularizer(self, w):
        sum = tf.constant(0.0, dtype="float32")

        mean = tf.reduce_mean(w, axis=1)
        for i in range(self.n_cultures):
            sum += tf.math.square(tf.norm(w[:, i] - mean))

        res = (self.lamb) * sum
        return res

    def get_best_idx(self, losses: list, cics: list, tau=0.15):
        tmp_losses = losses.copy()
        n_ls = math.ceil(len(losses) * tau)

        pairs = []
        tmp_cics = []

        for i in range(n_ls):
            val = min(tmp_losses)
            idx = tmp_losses.index(val)
            pairs.append((val, idx))
            tmp_losses.remove(val)
            tmp_cics.append(cics[idx])

        mincic = min(tmp_cics)
        for i in range(n_ls):
            if mincic == cics[i]:
                idx = i

        return pairs[i][1]
    
    def ModelSelection(
        self,
        TS,
        VS,
        aug,
        show_imgs=False,
        batches=[32],
        lrs=[1e-5, 1e-4, 1e-3],
        fine_lrs=[1e-6],
        epochs=[38],
        fine_epochs=12,
        nDropouts=[0.3],
        g=0.1,
        save=False,
        path="./"
    ):
        losses = []
        cics = []

    
        zipped_data = list(zip(*TS))
        random.shuffle(zipped_data)
        TS = tuple(map(list, zip(*zipped_data)))
        zipped_data = list(zip(*VS))
        random.shuffle(zipped_data)
        VS = tuple(map(list, zip(*zipped_data)))
        TS = (list(np.array(TS[0], dtype=np.float32)), TS[1])
        VS = (list(np.array(VS[0], dtype=np.float32)), VS[1])

        lambdas = np.logspace(-3, 0, 3)
        hyperparameters = []

        for lmb in lambdas:
            self.lamb = lmb
            for b in batches:
                for lr in lrs:
                  for ep in epochs:
                    for fine_lr in fine_lrs:
                        for nDropout in nDropouts:
                                self.model = None
                                history = self.DL(
                                    TS,
                                    VS,
                                    aug,
                                    show_imgs,
                                    b,
                                    lr,
                                    fine_lr,
                                    ep,
                                    fine_epochs,
                                    nDropout,
                                    g=g,
                                )
                                err_0 = 1-history.history["val_pred_dense_layer_0_accuracy"][-1]
                                err_1 = 1-history.history["val_pred_dense_layer_1_accuracy"][-1]
                                err_2 = 1-history.history["val_pred_dense_layer_2_accuracy"][-1]
                                loss = (err_0 + err_1 + err_2)/3

                                CIC = self.get_cic(VS[0], VS[1])
                                losses.append(loss)
                                cics.append(CIC)
                                hyperparameters.append({
                                    'batch_size': b,
                                    'lr': lr,
                                    'fine_lr': fine_lr,
                                    'nDropout': nDropout,
                                    'lambda': lmb,
                                    'epochs': ep
                                })
                                self.model = None
                                gc.collect()

        idx = self.get_best_idx(losses, cics)
        Hstar = hyperparameters[idx]
        best_loss = losses[idx]
        best_CIC = cics[idx]
        best_fine_lr = hyperparameters[idx]['fine_lr']
        best_lr = hyperparameters[idx]['lr']
        best_bs = hyperparameters[idx]['batch_size']
        best_lmb = hyperparameters[idx]['lambda']
        best_epochs = hyperparameters[idx]['epochs']
        best_nDropout = hyperparameters[idx]['nDropout']

        self.lamb = best_lmb
        TS = TS + VS
        self.DL(
            TS,
            None,
            aug,
            show_imgs,
            best_bs,
            best_lr,
            best_fine_lr,
            best_epochs,
            fine_epochs,
            best_nDropout,
            val=False,
            g=g,
        )

        if save:
                self.save(path)
  

    def DL(
        self,
        TS,
        VS,
        aug=False,
        show_imgs=False,
        batch_size=32,
        lr=1e-3,
        fine_lr=1e-5,
        epochs=1,
        fine_epochs=1,
        nDropout=0.2,
        g=0.1,
        val=True,
        
    ):
            shape = np.shape(TS[0][0])

            if show_imgs:
                images = []
                for i in range(9):
                    idx = np.random.randint(0, len(TS[0]) - 1)
                    images.append((TS[0][idx], TS[1][idx]))
                plt.figure(figsize=(10, 10))
                for i, (image, label) in enumerate(images):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(image)
                    plt.title(label)
                    plt.axis("off")
                plt.show()

            if val:
                monitor_val = "val_loss"
            else:
                monitor_val = "loss"

            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.01),
                    layers.GaussianNoise(g),
                    layers.RandomZoom(g, g),
                    layers.Resizing(shape[0], shape[1]),
                ]
            )
            seed = int(time.time() % 2**31)
            
                        
            if self.imbalanced:
                TS = self.ImbalancedTransformation(TS)
            
            if self.parify_batches_diffusion:
                Xb, yb = self.parify_batches_fast(TS, self.culture, batch_size)
                train_generator = tf.data.Dataset.from_tensor_slices((Xb, yb)) \
                                            .batch(batch_size) \
                                            .prefetch(tf.data.AUTOTUNE)
            else:
    
                train_generator = tf.data.Dataset.from_tensor_slices(
                            (tf.convert_to_tensor(TS[0]), tf.convert_to_tensor(TS[1]))
                        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            validation_generator = None
            if val:
                if self.parify_batches_diffusion:
                    Xb, yb = self.parify_batches_fast(VS, self.culture, batch_size)
                    
                    validation_generator = tf.data.Dataset.from_tensor_slices((Xb, yb)) \
                                            .batch(batch_size) \
                                            .prefetch(tf.data.AUTOTUNE)
                else:
                    validation_generator = tf.data.Dataset.from_tensor_slices(
                            (tf.convert_to_tensor(VS[0]), tf.convert_to_tensor(VS[1]))
                        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            if aug:
                if show_imgs:
                    idx = np.random.randint(0, len(TS) - 1)
                    images = []
                    images.append((TS[0][idx], TS[1][idx]))
                    for ims, labels in images:
                        plt.figure(figsize=(10, 10))
                        for i in range(9):
                            ax = plt.subplot(3, 3, i + 1)

                            augmented_image = data_augmentation(
                                tf.expand_dims(ims, 0), training=True
                            )
                            plt.imshow(augmented_image[0].numpy().astype("int32"))
                            plt.title(int(labels))
                            plt.axis("off")
                        plt.show()
            base_model = keras.applications.ResNet50V2(
                weights="imagenet", 
                input_shape=shape,
                include_top=False,
            )  
            base_model.trainable = False
            inputs = keras.Input(shape=shape)
            scale_layer = keras.layers.Rescaling(scale=1 / 255.0)
            if aug:
                x = data_augmentation(inputs)   
                x = scale_layer(x)
            else:
                x = scale_layer(inputs)

            
            x = base_model(x, training=False)
            y = keras.layers.GlobalAveragePooling2D()(x)
            y = keras.layers.Flatten()(y)
            outputs = []
            for i in range(self.n_cultures):
                outputs.append(keras.layers.Dense(1, activation='sigmoid', name=f'pred_dense_layer_{i}')(y))
            
            self.model = keras.Model(inputs, outputs = outputs)


            lr_reduce = ReduceLROnPlateau(
                monitor=monitor_val,
                factor=0.2,
                patience=5,
                verbose=self.verbose_param,
                min_lr=1e-9,
            )
            early = EarlyStopping(
                monitor=monitor_val,
                min_delta=0.001,
                patience=10,
                verbose=self.verbose_param,
                mode="auto",
            )
            callbacks = [early, lr_reduce]

            
            self.model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss=self.custom_loss(),
                metrics=self.custom_accuracy(),
            )
            self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
                shuffle=True
            )
            base_model.trainable = True

            self.model.compile(
                optimizer=keras.optimizers.Adam(fine_lr),  
                loss=self.custom_loss(),
                metrics=self.custom_accuracy(),
            )

            history = self.model.fit(
                train_generator,
                epochs=fine_epochs,
                validation_data=validation_generator,
                verbose=self.verbose_param,
                callbacks=callbacks,
                shuffle=True
            )
            tf.keras.backend.clear_session()
            return history

    def fit(
        self,
        TS,
        VS=None,
        out_dir="./",
        save=False,
        aug=0,
        g=0.1,
    ):
        
        if self.type == "SVC":
            self.SVC(TS)
        elif self.type == "RFC":
            self.RFC(TS)
        elif self.type == "DL" or "RESNET":
            self.ModelSelection(TS, VS, aug=aug, g=g, save=save, path=out_dir)
        else:
            self.ModelSelection(TS, VS, aug=aug, g=g, save=save, path=out_dir)

    def get_model_from_weights(self, size, eps=0.05, mult=0.2, path="./"):
        self.model = tf.keras.models.load_model(path)

def random_culture(n_cultures, culture):
    choices = [i for i in range(n_cultures) if i != culture]  
    return random.choice(choices) if choices else None   

class ProcessingClass:
    def __init__(
        self, shallow, lamp, memory_limit=2700, basePath="./" 
    ) -> None:
        strObj = DeepStrings()
        if lamp:
            paths = strObj.lamp_paths
        else:
            paths = strObj.carpet_paths_str
    
        if paths:
            self.dataobj = DataClass(paths)
        else:
            raise Exception("Carpet Problem has not been tackled in shallow learning")
        self.shallow = shallow
        self.lamp = lamp
        self.basePath = basePath
    def parify_batches(self, s, culture, hot_encoding, size):
        indeces_per_culture = []
        batch_size = 64
        if hot_encoding:
            for i in range(self.n_cultures):
                c = np.zeros(self.n_cultures)
                c[i] = 1.0
                vals = (np.where((np.asarray(s[1], dtype=object)[:, :self.n_cultures] == c).all(axis=1))[0])
                if i == culture:
                    n_samples_majority = len(vals)
                indeces_per_culture.append(np.where((np.asarray(s[1], dtype=object)[:, :self.n_cultures] == c).all(axis=1))[0])
        else:
            n_samples_majority = len(np.where(np.asarray(s[1], dtype=object)[self.n_cultures]==culture))
            for i in range(self.n_cultures):
                indeces_per_culture.append([np.where(np.asarray(s[1], dtype=object)[:,1])==i])
        B = []
        Blabel = []
        DS = []
        DSlabel = []
        for i in range(n_samples_majority):
            for j in range(self.n_cultures):
                if len(B)>= batch_size:
                    DS.append(np.asarray(B))
                    B = []
                    DSlabel.append(np.asarray(Blabel))
                    Blabel = []
                sample = np.asarray(s[0])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                label = np.asarray(s[1])[indeces_per_culture[j][i % len(indeces_per_culture[j])]]
                B.append(cv2.resize(sample, (size, size), interpolation = cv2.INTER_CUBIC))
                Blabel.append(label)
        if len(B)>0:
            for i in range(batch_size-len(B)):
                j = i % self.n_cultures
                rnd_index = np.random.randint(0, len(indeces_per_culture[j]))
                B.append(s[0][indeces_per_culture[j][rnd_index]])
                Blabel.append(s[1][indeces_per_culture[j][rnd_index]])
            DS.append(np.asarray(B))
            DSlabel.append(np.asarray(Blabel))
        return DS, DSlabel
    def prepare_data(
        self,
        culture,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.2,
        n: int = 1000,
        augment=0,
        gaug = 0.01,
        imbalanced=0,
        discriminator=0,
        diffusion = 0,
        aug = 0,
        weights = 0,
        only_minority_diffusion=0,
        parify_batches_diffusion=0,
        plt_imgs = False
    ):
        self.dataobj.prepare(
            culture=culture,
            percent=percent,
            shallow=self.shallow,
            val_split=val_split,
            test_split=test_split,
            n=n,
            imbalanced=imbalanced
        )
        if augment:
            prepObj = PreprocessingClass()
            X_augmented = prepObj.classical_augmentation(
                X=self.dataobj.X, g=gaug, 
            )
            self.dataobj.X.extend(X_augmented)
            self.dataobj.y.extend(self.dataobj.y)
        if diffusion==1 and not discriminator:
            size = 100
            n_imgs = len(self.dataobj.X)//4
            diff_model = DiffusionStandardModel(image_size=size)
            init_shape = np.shape(self.dataobj.X[0])[0:2]
            bpath = "./"
            bpath = bpath + '/MIT/'
            if parify_batches_diffusion:
                bpath = bpath + '/PAR_BS/'
            if only_minority_diffusion:
                bpath = bpath + '/ONLY_MIN/'
            fObj = FileManagerClass(bpath)
            fObj = FileManagerClass(bpath+'/GeneratedImages/')
            if parify_batches_diffusion:
                    for j in range(2):
                            tempX = [
                                cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j
                            ]
                            tempXv = [
                                cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j
                            ]    
                            tempY = [
                                self.dataobj.y[i]
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j
                            ]
                            tempYv = [
                                self.dataobj.yv[i]
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j
                            ]     
                            tempX, _ = self.parify_batches((tempX, tempY), culture, True, size)
                            tempXv, _ = self.parify_batches((tempXv, tempYv), culture, True, size)               
                            images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs,  plot_imgs = plt_imgs, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion,  base_path=bpath, onlymin=only_minority_diffusion)
                            for img in images:
                                img = np.asarray(img)
                                img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                img = np.asarray(img, dtype=np.float32)
                                self.dataobj.X.append(img)
                                c = random_culture(self.n_cultures, culture)
                                lbl = list(np.zeros(self.n_cultures))
                                lbl[c]=1.0
                                lbl.append(j)
                                self.dataobj.y.append(lbl)
            else: 
                if only_minority_diffusion:
                    for j in range(2):
                            
                            tempX = [
                                cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j and np.argmax(self.dataobj.y[i][0:self.n_cultures])==culture
                            ]
                            tempXv = [
                                cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j and np.argmax(self.dataobj.y[i][0:self.n_cultures])==culture
                            ]                    
                            images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs,  plot_imgs = plt_imgs, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion, base_path=bpath, onlymin=only_minority_diffusion)
                            for img in images:
                                img = np.asarray(img)
                                img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                img = np.asarray(img, dtype=np.float32)
                                self.dataobj.X.append(img)
                                c = random_culture(self.n_cultures, culture)
                                lbl = list(np.zeros(self.n_cultures))
                                lbl[c]=1.0
                                lbl.append(j)
                                self.dataobj.y.append(lbl)
                    
                else:
                    for j in range(2):
                            for i in range(len(self.dataobj.X)):
                             pass
                            tempX = [
                                cv2.resize(self.dataobj.X[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.X))
                                if self.dataobj.y[i][self.n_cultures]== j
                            ]
                            tempXv = [
                                cv2.resize(self.dataobj.Xv[i], (size, size), interpolation = cv2.INTER_CUBIC)
                                for i in range(len(self.dataobj.Xv))
                                if self.dataobj.yv[i][self.n_cultures] == j
                            ]                    
                            images = diff_model.learn_on_custom_dataset(tempX, tempXv, n_images = n_imgs,  plot_imgs = plt_imgs, aug=aug, percent=percent, lamp=self.lamp, culture=culture, category=j, imb=imbalanced, parify_batches_diffusion=parify_batches_diffusion, base_path=bpath, onlymin=only_minority_diffusion)
                            for img in images:
                                img = np.asarray(img)
                                img = cv2.resize(img,  init_shape, interpolation = cv2.INTER_CUBIC)
                                img = np.asarray(img, dtype=np.float32)
                                self.dataobj.X.append(img)
                                c = random_culture(self.n_cultures, culture)
                                lbl = list(np.zeros(self.n_cultures)) 
                                lbl[c]=1.0
                                lbl.append(j)
                                self.dataobj.y.append(lbl)
        

    def prepare_test(
        self,
        augment=0,
        g_rot: float = 0.1,
        g_noise: float = 0.1,
        g_bright: float = 0.1,
        culture=None,
        nt=None,
    ):
        
        self.Xt_totaug = []
        self.Xt_aug = []
        if nt != None and nt < len(self.dataobj.Xt):
            self.dataobj.Xt = self.dataobj.Xt[0:nt]
        for culture in range(3):
            if augment:
                        prepObj = PreprocessingClass()
                        self.Xt_aug.append(
                            prepObj.classical_augmentation(
                                X=self.dataobj.Xt[culture],
                                g_rot=g_rot,
                                g_noise=g_noise,
                                g_bright=g_bright,
                            )
                        )
            
                

    def process(
        self,
        type="DL",
        verbose_param=0,
        learning_rate=0.001,
        epochs=15,
        batch_size=2,
        lambda_index=0,
        culture=0,
        percent=0,
        val_split: float = 0.2,
        test_split: float = 0.1,
        n: int = 1000,
        augment=0,
        gaug: float = 0.1,
        discriminator=0,
        eps=0.3,
        gradcam=False,
        complete=0,
        n_cultures=3,
        imbalanced=0,
        diffusion=0,
        only_minority_diffusion=0,
        parify_batches_diffusion=0,
    ):
        weights = np.ones(n_cultures) * 1/3 #percent
        weights[culture] = 1  # this are the proportions in the dataset
        self.n_cultures = n_cultures
        self.prepare_data(
            culture=culture,
            percent=percent,
            val_split=val_split,
            test_split=test_split,
            n=n,
            augment=augment,
            imbalanced=imbalanced,
            discriminator=discriminator,
            diffusion = diffusion,
            gaug = gaug,
            weights = weights,
            only_minority_diffusion=only_minority_diffusion,
            parify_batches_diffusion=parify_batches_diffusion
        )
        self.model = None
    
        self.basePath = self.basePath + "MIT/" + type

        if imbalanced:
            self.basePath = self.basePath + "/IMB/"
        else:
            self.basePath = self.basePath + "/BAL/"
        if self.lamp:
            if culture == 0:
                c = "/LC/"
            elif culture == 1:
                c = "/LF/"
            elif culture == 2:
                c = "/LT/"
            else:
                c = "/LC/"
        else:
            if culture == 0:
                c = "/CI/"
            elif culture == 1:
                c = "/CJ/"
            elif culture == 2:
                c = "/CS/"
            else:
                c = "/CI/"
        self.basePath = self.basePath + c + str(percent) + "/"
        if diffusion: 
            self.basePath = self.basePath + "DIFFUSION/"
            if only_minority_diffusion:
                self.basePath = self.basePath + "ONLY_MIN/"
        if parify_batches_diffusion:
                self.basePath = self.basePath + "PAR_BS/"
        if augment:
 
                aug = f"STDAUG/g={gaug}/"
        else:
            
                aug = "NOAUG/"

        self.basePath = self.basePath + aug
        if (not complete):
            self.basePath = self.basePath + str(lambda_index) + "/"

        
        self.model = MitigatedModelsAdvanced(
            type=type,
            culture=culture,
            verbose_param=verbose_param,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lambda_index=lambda_index,
            n_cultures=n_cultures,
            imbalanced=imbalanced,
            diffusion=diffusion,
            weights=weights,
            parify_batches_diffusion=parify_batches_diffusion
        )
                

        self.model.fit(
            (self.dataobj.X, self.dataobj.y),
            (self.dataobj.Xv, self.dataobj.yv),
            eps=eps,
            gradcam=gradcam,
            out_dir=self.basePath,
            complete=complete,
            aug=augment,
            g=gaug,
        )
        
        self.imbalanced = imbalanced


    
    def test(
        self,
        culture=0,
        augment=0,
        gaug=0.1,
        eps=0.3,
        nt=None,
        discriminator=0,
    ):
        
        if self.model:
            self.prepare_test(
                augment=augment,
                g_rot=gaug,
                g_noise=gaug,
                g_bright=gaug,
                culture=culture,
                eps=eps,
            )
        else:
            return -1
        for culture in range(3):
                for i in range(3):
                        if augment:
                            
                                cm = self.model.get_model_stats(
                                    self.Xt_aug[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TSTDAUG/G_AUG={gaug}/"
                        else:
                            
                                cm = self.model.get_model_stats(
                                    self.dataobj.Xt[culture],
                                    self.dataobj.yt[culture],
                                    i,
                                    discriminator=discriminator,
                                )
                                testaug = f"TNOAUG/"
                        testaug = testaug + f"CULTURE{culture}/"
                        path = self.basePath + testaug + "out " + str(i) + ".csv"
                        self.save_results(cm, path, discriminator=discriminator)
        
        return

    def save_results(self, cm, path, discriminator=0):
        
        fObj = FileManagerClass(path)
        fObj.writecm(cm, discriminator=discriminator)

    def partial_clear(self, basePath=None):
        self.model = None
        self.dataobj.clear()
        self.Xt_totaug = None
        self.Xt_aug = None
        self.basePath = basePath

        gc.collect()

class AdamW(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate=0.001, weight_decay=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7, **kwargs):
        super(AdamW, self).__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, **kwargs)
        self.weight_decay = weight_decay  

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        
        grads_and_vars = [
            (grad + self.weight_decay * var, var) if grad is not None else (None, var)
            for grad, var in grads_and_vars
        ]
        
        return super(AdamW, self).apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({'weight_decay': self.weight_decay})
        return config

dataset_name = "places365_small"
dataset_repetitions = 5
num_epochs = 75  
num_epochs_flowers = 1

kid_image_size = 75
kid_diffusion_steps = 15
plot_diffusion_steps = 15
min_signal_rate = 0.02
max_signal_rate = 0.95
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 100, 128]
block_depth = 2
batch_size = 64
ema = 0.999
transfer_learning_rate = 1e-3
learning_rate = 1e-3
weight_decay = 1e-4

attention_type = "self"

def AttentionBlock(width):
    def apply(x):
        query = layers.Conv2D(width, kernel_size=1)(x)
        key = layers.Conv2D(width, kernel_size=1)(x)
        value = layers.Conv2D(width, kernel_size=1)(x)
        attention_scores = tf.keras.layers.Attention()([query, key])
        attention_output = layers.Multiply()([attention_scores, value])
        return layers.Add()([x, attention_output])
    
    return apply

def SelfAttentionBlock(channels):
    
    def apply(x):
        query = layers.Conv2D(channels // 8, kernel_size=1)(x)
        key = layers.Conv2D(channels // 8, kernel_size=1)(x)
        value = layers.Conv2D(channels, kernel_size=1)(x)
        attention_scores = tf.nn.softmax(tf.matmul(
            tf.reshape(query, [tf.shape(x)[0], -1, channels // 8]),  
            tf.reshape(key, [tf.shape(x)[0], -1, channels // 8]), transpose_b=True
        ))
        attention_output = tf.matmul(attention_scores, 
                                      tf.reshape(value, [tf.shape(x)[0], -1, channels]))
        attention_output = tf.reshape(attention_output, tf.shape(x))
        return layers.Add()([x, attention_output])  

    return apply

def SEBlock(channels, reduction=16):
    
    def apply(x):
        squeeze = layers.GlobalAveragePooling2D()(x)
        squeeze = layers.Dense(channels // reduction, activation="relu")(squeeze)
        squeeze = layers.Dense(channels, activation="sigmoid")(squeeze)
        scale = layers.Reshape((1, 1, channels))(squeeze)
        return layers.Multiply()([x, scale])

    return apply

def TransformerBlock(channels, num_heads=4, ff_dim=256):

    def apply(x):
        x_norm = layers.LayerNormalization()(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels)(x_norm, x_norm)
        attention_output = layers.Add()([x, attention_output])  
        ff_output = layers.Dense(ff_dim, activation="relu")(attention_output)
        ff_output = layers.Dense(channels, name="transformer_dense")(ff_output)
        return layers.Add()([attention_output, ff_output])  

    return apply

def preprocess_image(image_size = 128):
    def  preprocess_function(data):
        height = tf.shape(data["image"])[0]
        width = tf.shape(data["image"])[1]
        crop_size = tf.minimum(height, width)
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )
        image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return preprocess_function

def prepare_dataset(split, image_size = 128, add_to_ds = None):
    data = tfds.load(dataset_name, split=split, shuffle_files=False)
    data = tf.data.Dataset.from_tensor_slices(list(data.map(preprocess_image(image_size), num_parallel_calls=tf.data.AUTOTUNE)))

    if add_to_ds!=None:
        data = data.concatenate(add_to_ds)
    data = data.cache().repeat(dataset_repetitions).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    return data
class KID(tf.keras.metrics.Metric):
    def __init__(self, name, image_size, **kwargs):
        super().__init__(name=name, **kwargs)
        self.kid_tracker = tf.keras.metrics.Mean(name="kid_tracker")
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
                tf.keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype="float32")
        return (
            features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0
        ) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)
        batch_size = real_features.shape[0]
        batch_size_f = tf.cast(batch_size, dtype="float32") 
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace( 
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def ResidualBlock(width, activation="relu", normalization="batch"):
    def apply(x):
        input_width = x.shape[-1]
        residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)

        if normalization == "batch":
            x = layers.BatchNormalization()(x)
        elif normalization == "layer":
            x = layers.LayerNormalization()(x)

        x = layers.Activation(activation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

        x = layers.Add()([x, residual])
        return x

    return apply

def DownBlock(width, block_depth, pool_type="average", dropout_rate=0.1):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)

        if pool_type == "average":
            x = layers.AveragePooling2D(pool_size=2)(x)
        elif pool_type == "max":
            x = layers.MaxPooling2D(pool_size=2)(x)

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        return x

    return apply

def fix_shape_mismatch(x, skip):
    
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        if x.shape[1] > skip.shape[1] or x.shape[2] > skip.shape[2]:
            dim = (float(x.shape[1]) - float(skip.shape[1])) / 2.0
            dim1 = math.ceil(dim)
            dim2 = math.floor(dim)
            cropping = (dim1, dim2)
            x = layers.Cropping2D(((cropping, cropping)))(x)
            return x
        else:
            dim = (float(skip.shape[1]) - float(x.shape[1])) / 2.0
            dim1 = math.ceil(dim)
            dim2 = math.floor(dim)
            padding = (dim1, dim2)
            x = layers.ZeroPadding2D(((padding, padding)))(x)
            return x
    return x

def UpBlock(width, block_depth, dropout_rate=0.1):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            skip = skips.pop()
            x = fix_shape_mismatch(x, skip)
            x = layers.Concatenate()([x, skip])
            x = ResidualBlock(width)(x)
            if dropout_rate:
                x = layers.Dropout(dropout_rate)(x)
        return x

    return apply

def get_network(image_size, widths, block_depth, attention_type="transformer", pool_type="average", dropout_rate=0.1):

    noisy_images = tf.keras.Input(shape=(image_size, image_size, 3))
    noise_variances = tf.keras.Input(shape=(1, 1, 1))
    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    data_augmentation = keras.Sequential(
                    [
                        layers.Rescaling(1.0/255.0),
                        layers.RandomFlip("horizontal"),
                        layers.RandomRotation(0.05),
                        layers.GaussianNoise(0.0001),
                        layers.Rescaling(255.0),
                    ]
                )
    noisy_images = data_augmentation(noisy_images)
    
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth, pool_type=pool_type, dropout_rate=dropout_rate)([x, skips])

    
    for i in range(block_depth):
        if i == block_depth // 2: 
            if attention_type == "self":
                x = SelfAttentionBlock(widths[-1])(x)
            elif attention_type == "transformer":
                x = TransformerBlock(widths[-1])(x)
        x = ResidualBlock(widths[-1])(x)
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth, dropout_rate=dropout_rate)([x, skips])

    x = layers.Conv2D(3, kernel_size=1)(x)

    return tf.keras.Model([noisy_images, noise_variances], x, name="attention_residual_unet")
class DiffusionStandardModel(tf.keras.Model):
    def __init__(self, image_size, widths=widths, block_depth=block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = tf.keras.models.clone_model(self.network)
        self.image_size = image_size

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid", image_size=self.image_size)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.cast(tf.math.acos(max_signal_rate), "float32")
        end_angle = tf.cast(tf.math.acos(min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images, diffusion_steps):
        initial_noise = tf.random.normal(
            shape=(num_images, self.image_size, self.image_size, 3)
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, 3))
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  
            image_loss = self.loss(images, pred_images)  

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, 3))
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        def close_event():
            plt.close()
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        timer = fig.canvas.new_timer(interval = 2000)
        timer.add_callback(close_event)
        timer.start()
        plt.savefig(self.img_name)
        plt.show()
        plt.close()

    def plot_dataset(self, ds, num_rows=3, num_cols=6):
        def close_event():
            plt.close()
        fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(ds[index])
                plt.axis("off")
        plt.tight_layout()
        timer = fig.canvas.new_timer(interval = 3000) 
        timer.add_callback(close_event)
        timer.start()
        plt.show()
        plt.close()  


    def learn_on_custom_dataset(self, train_dataset, val_dataset, n_images = 100, plot_imgs = False, aug=False, save=True, get_pretrained=True, percent=0.0, lamp=False, culture=0, category=0, imb=0, model_selection=True, parify_batches_diffusion=0, base_path = './', onlymin=0): 
        
        if aug:
            suppress_output()
            data_augmentation = keras.Sequential(
                    [
                        layers.Rescaling(1.0/255.0),
                        layers.RandomFlip("horizontal"),
                        layers.GaussianNoise(0.01),
                        layers.Rescaling(255.0),
                    ]
                )
                
            for i in range(5):
                aug_images = data_augmentation(train_dataset)
                for img in aug_images:
                    img = tf.clip_by_value(img, 0, 255)
                    img = tf.cast(img, "uint8")
                    train_dataset.append(img)

        if not get_pretrained:
            early = EarlyStopping(
                monitor="val_kid",
                min_delta=0.001,
                patience=7,
            )
            lr_reduce = ReduceLROnPlateau(
                monitor="val_kid",
                factor=0.1,
                patience=3,
                verbose=1,
                min_lr=1e-9) 
            callbacks = [early, lr_reduce]  
            tf_lr = transfer_learning_rate
            for i in range(23):
                flowers_dataset = prepare_dataset(f"train[{3.0*(i)}:{3.0*(i+1)}%]+test[{3.0*(i)}:{3.0*(i+1)}%]", image_size=self.image_size)
                val_flowers_dataset = prepare_dataset(f"train[{100-(1.0)*(i+1)}%:{100-(1.0)*(i)}]+test[{100-(1.0)*(i+1)}%:{100-(1.0)*(i)}]", image_size=self.image_size)
                self.normalizer.adapt(flowers_dataset)
                
                
                self.compile(
                        optimizer=AdamW(
                            learning_rate=tf_lr, weight_decay=weight_decay
                        ),
                        loss=tf.keras.losses.mean_absolute_error,
                    )
                
                self.img_name = base_path + "/PretrainedNetGeneration.png"
                self.fit(
                    flowers_dataset,
                    epochs=num_epochs_flowers,
                    validation_data=val_flowers_dataset,
                    callbacks=callbacks,
                    shuffle=True
                )
                tf_lr = tf_lr / 1.4
                self.network.save( base_path +'diffusion_pretrained.h5')
                self.ema_network.save( base_path +'ema_diffusion_pretrained.h5')
        train_dataset = tf.data.Dataset.from_tensor_slices(list(np.asarray(train_dataset, dtype="float32") / 255.0))
        val_dataset = tf.data.Dataset.from_tensor_slices(list(np.asarray(val_dataset, dtype="float32") / 255.0))
        if parify_batches_diffusion==0:
            
            TS = train_dataset.batch(batch_size, drop_remainder=True)
            VS = val_dataset.batch(batch_size, drop_remainder=True)
        else:
            TS = train_dataset
            VS = val_dataset
        
        self.normalizer.adapt(train_dataset)

        early = EarlyStopping(
                monitor="val_kid",
                min_delta=0.001,
                patience=15,
        )
        lr_reduce = ReduceLROnPlateau(
                monitor="val_kid",
                factor=0.2,
                patience=6,
                verbose=1,
                min_lr=1e-9,
            )

        callbacks = [early, lr_reduce]
        if model_selection:
            best_kid = np.inf
            for ep in [30, 50]:
                for l_r in np.logspace(-5, -3, 3):
                    self.network = tf.keras.models.load_model('diffusion_pretrained.h5')
                    self.ema_network = tf.keras.models.load_model('ema_diffusion_pretrained.h5')
                    self.compile(
                            optimizer=AdamW(
                                learning_rate=l_r, weight_decay=weight_decay
                            ),
                            loss=tf.keras.losses.mean_absolute_error,
                        )
                    self.network.save( base_path +'/new/diffusion_pretrained.tf')
                    self.ema_network.save( base_path +'/new/ema_diffusion_pretrained.tf')

                    for layer in self.network.layers[0:int(len(self.network.layers)/2)]:
                        layer.trainable = False
                    for layer in self.ema_network.layers[0:int(len(self.ema_network.layers)/2)]:
                        layer.trainable = False

                    
                    self.img_name =  base_path +"/PretrainedNetGeneration.png"
                    self.plot_images()

                    if lamp:
                        self.img_name =  base_path + f"/GeneratedImages/{percent}/Lamps{culture}_{category}_imb={imb}.png"
                    else:
                        self.img_name =  base_path +f"GeneratedImages/{percent}/Carpets{culture}_{category}_imb={imb}.png"
                    fObj = FileManagerClass(self.img_name)
                    history = self.fit(
                        TS,
                        epochs=ep,
                        validation_data=VS,
                        callbacks=callbacks,
                        shuffle=True
                    )
                    kid = history.history["val_kid"][-1]
                    if kid < best_kid:
                        best_kid = kid
                        best_epochs = ep
                        best_lr = l_r
        else:
            best_epochs = num_epochs
            best_lr = learning_rate
        self.network = tf.keras.models.load_model('diffusion_pretrained.h5')
        self.ema_network = tf.keras.models.load_model('ema_diffusion_pretrained.h5')
        self.compile(
                optimizer=AdamW(
                    learning_rate=best_lr, weight_decay=weight_decay
                ),
                loss=tf.keras.losses.mean_absolute_error,
            )
            
        for layer in self.network.layers[0:int(len(self.network.layers)/2)]:
            layer.trainable = False
        for layer in self.ema_network.layers[0:int(len(self.ema_network.layers)/2)]:
            layer.trainable = False

 

        if lamp:
            self.img_name =  base_path +f"/GeneratedImages/{percent}/Lamps{culture}_{category}_imb={imb}.png"
        else:
            self.img_name =  base_path +f"/GeneratedImages/{percent}/Carpets{culture}_{category}_imb={imb}.png"

        fObj = FileManagerClass(self.img_name)
        self.fit(
            TS,
            epochs=best_epochs,
            validation_data=VS,
            callbacks=callbacks,
            shuffle=True
        )
        
        tot = 0
        generated_images = []
        for i in range(n_images//batch_size +1):
            tot +=batch_size
            n_ = min((n_images-tot), batch_size)
            if n_>0:
                images = self.generate(
                    num_images=n_,
                    diffusion_steps=plot_diffusion_steps,
                )
                for img in images:
                    generated_images.append(img*255)

        generated_images = np.asarray(generated_images)

        self.network.save( base_path + f'{percent}/Lamps{culture}_{category}_imb={imb}' +'diffusion_pretrained.h5')
        self.ema_network.save( base_path + f'{percent}/Lamps{culture}_{category}_imb={imb}' +'ema_diffusion_pretrained.h5')

        return generated_images
 
    def plot_examples(self, base_path, culture, category, imb, diffusion_steps=kid_diffusion_steps):
        pt =  base_path +f"/GeneratedImages/Carpets{culture}_{category}_imb={imb}/"
        fObj = FileManagerClass(pt)
        num_images = 1
        initial_noise = tf.random.normal(
        shape=(num_images, self.image_size, self.image_size, 3)
        )
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise
        for trial in range(5):
        
            for step in range(diffusion_steps):
                noisy_images = next_noisy_images
                diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
                noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
                pred_noises, pred_images = self.denoise(
                    noisy_images, noise_rates, signal_rates, training=False
                )
                next_diffusion_times = diffusion_times - step_size
                next_noise_rates, next_signal_rates = self.diffusion_schedule(
                    next_diffusion_times
                )
                next_noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
                )
                imgs_to_plot = self.denormalize(pred_images)



memory_limit = 9500
percents = [0.05]
verbose_param = 1
n = 1000
class_divisions = [ 0]
imbalances = [0]
g_gaugs = np.logspace(-4, -1, 4)
eps = np.logspace(-3, -1, 3)
g_aug = g_gaugs[0]
cs = [ 0, 1, 2]
lamps = [0, 1]
ep = eps[0]
imb = 0
diffusions = [1, 0]
ks = [0, 1]
parify_batches_diffusions = [0, 1]
basePath = "./try2/"
for percent in percents:
 for parify_batches_diffusion in parify_batches_diffusions:
  for k in ks:
    for lamp in lamps:
     for diffusion in diffusions:
      if diffusion and not k:
        break
      else:
        for c in cs:
            for cl_div in class_divisions:
                procObj = ProcessingClass(
                    shallow=0,
                    lamp=lamp,
                    memory_limit=memory_limit,
                    basePath=basePath,
                )
                model = None
                procObj.process(
                    type="DL",
                    verbose_param=verbose_param,
                    culture=c,
                    percent=percent,
                    n=n,
                    augment=k % 2,
                    gaug=g_aug,
                    eps =ep,
                    class_division=cl_div,
                    imbalanced=imb, 
                    diffusion = diffusion,
                    only_minority_diffusion=1,
                    parify_batches_diffusion=parify_batches_diffusion,
                    mitigation_type=1
                )
                procObj.test(
                    culture=c,
                    augment=0,
                    gaug=0,
                )
                procObj.partial_clear(basePath)
