from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import cv2
from tf_explain.utils.display import grid_display, heatmap_display
from tf_explain.utils.saver import save_rgb


class GeneralModelClass:

    def __init__(self) -> None:
        self.model = 0

    def __call__(self, X):
        if self.model != None:
            with tf.device("/gpu:0"):
                if tf.is_tensor(X):
                    yP = self.model(X)
                    if tf.shape(yP)[0] > 1:
                        return tf.gather(yP, indices=[[0, 0], [1, 0], [2, 0]])
                    else:
                        return yP[0][0]
                else:
                    yP = np.asarray(self.model(X))
                    if type(self.model) == keras.engine.functional.Functional:
                        if tf.shape(yP)[0] > 1:
                            # print(np.shape(yP))
                            return yP[:, 0]
                        else:
                            # print(np.shape(yP))
                            return yP[0]
                    else:
                        return yP
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
        return values

    def test(self, Xt, out=-1):
        if self.model:
            yF = []
            for xt in Xt:
                if out < 0:
                    yF.append(np.asarray(self(xt[None, ...])))
                else:
                    # print(np.shape(self.model(xt[None, ...])))
                    pL = np.asarray(self(xt[None, ...]))[out]
                    yF.append(pL)
                    #print(f"predicted label is {pL}")
                    #plt.imshow(xt)
                    #plt.show()
            yFq = self.quantize(yF)
            return yFq
        else:
            print("Try fitting the model before")
            return None

    def get_model_stats(self, Xt, yT, out=-1):
        yFq = self.test(Xt, out)
        if len(np.shape(yT)) > 1:
            if type(yT) == np.ndarray:
                yT = yT[:, 1]
            elif type(yT) == list:
                yT = np.asarray(yT)[:, 1]
        
        if yFq:
            # yT = list([c_i, y_i])
            cm = confusion_matrix(y_true=yT, y_pred=yFq)
            return cm
        



    def explain(
        self,
        validation_data,
        class_index,
        layer_name=None,
        use_guided_grads=True,
        colormap=cv2.COLORMAP_VIRIDIS,
        image_weight=0.7,
    ):
        """
        Compute GradCAM for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            layer_name (str): Targeted layer for GradCAM. If no layer is provided, it is
                automatically infered from the model architecture.
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating the weight of
                the input image to be overlaying the calculated attribution maps. Defaults to `0.7`.
            use_guided_grads (boolean): Whether to use guided grads or raw gradients

        Returns:
            numpy.ndarray: Grid of all the GradCAM
        """
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
        """
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.

        Args:
            model (tf.keras.Model): tf.keras model to inspect

        Returns:
            str: Name of the target layer
        """
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
        """
        Generate guided gradients and convolutional outputs with an inference.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int): Index of targeted class
            use_guided_grads (boolean): Whether to use guided grads or raw gradients

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output]
        )
        print(f"layer name is {layer_name}")
        print(f"Created grad model with:\n inputs:{self.model.inputs};\n output:{[self.model.get_layer(layer_name).output, self.model.output]}")
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            print(f"input in get_gradients and filters: {np.shape(inputs)}")
            conv_outputs, predictions = grad_model(inputs)
            print(f"conv_outputs in get_gradients and filters: {np.shape(conv_outputs)}")
            print(f"predictions in get_gradients and filters: {np.shape(predictions)}")
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
        """
        Apply Grad CAM algorithm scheme.

        Inputs are the convolutional outputs (shape WxHxN) and gradients (shape WxHxN).
        From there:
            - we compute the spatial average of the gradients
            - we build a ponderated sum of the convolutional outputs based on those averaged weights

        Args:
            output (tf.Tensor): Target layer outputs, with shape (batch_size, Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (batch_size, Hl, Wl, Nf)

        Returns:
            List[tf.Tensor]: List of ponderated output of shape (batch_size, Hl, Wl, 1)
        """
        print(f"outputs inside function generated_ponderated_output = {np.shape(outputs)}")
        print(f"grads inside function generated_ponderated_output = {np.shape(grads)}")
        maps = [
            self.ponderate_output(output, grad)
            for output, grad in zip(outputs, grads)
        ]

        return maps

    #@staticmethod
    def ponderate_output(self, output, grad):
        """
        Perform the ponderation of filters output with respect to average of gradients values.

        Args:
            output (tf.Tensor): Target layer outputs, with shape (Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (Hl, Wl, Nf)

        Returns:
            tf.Tensor: Ponderated output of shape (Hl, Wl, 1)
        """

        print(f"output inside function ponderated_output = {output}")
        print(f"grad inside function ponderated_output = {grad}")

        weights = tf.reduce_mean(grad, axis=(0, 1))

        # Perform ponderated sum : w_i * output[:, :, i]
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
