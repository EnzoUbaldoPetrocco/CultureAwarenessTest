#!/usr/bin/env python
__author__ = "Enzo Ubaldo Petrocco"
# import the necessary packages
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt


class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()
			
	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
	
	def compute_heatmap(self, images, eps=1e-8, out = 0, path='./'):
		with tf.device("/gpu:0"):
			# construct our gradient model by supplying (1) the inputs
			# to our pre-trained model, (2) the output of the (presumably)
			# final 4D layer in the network, and (3) the output of the
			# softmax activations from the 
			#self.model.summary()
			
			for i, image in enumerate(images):
				
				gradModel = tf.keras.Model(
					inputs=[self.model.layers[0].input],
					outputs=[self.model.get_layer('conv5_block3_out').output, self.model.output])
				
				# record operations for automatic differentiation
				with tf.GradientTape() as tape:
					# cast the image tensor to a float-32 data type, pass the
					# image through the gradient model, and grab the loss
					# associated with the specific class index
					inputs = tf.cast(image, tf.float32)
					#convOutputs = gradModel(inputs)
					(convOutputs, predictions) = gradModel(np.expand_dims(inputs, 0))
					#predictions = self.model(inputs)
					#print(f"Predictions shape={np.shape(predictions)}")
					if (out<0):
						loss = predictions[:, self.classIdx]
					else:
						if len(predictions)>1:
							loss = predictions[out][0]
						else:
							loss = predictions[0][out]
							#print(f"Shape of loss= {np.shape(loss)}")

				# use automatic differentiation to compute the gradients
				grads = tape.gradient(loss, convOutputs)

				# compute the guided gradients
				castConvOutputs = tf.cast(convOutputs > 0, "float32")
				castGrads = tf.cast(grads > 0, "float32")
				guidedGrads = castConvOutputs * castGrads * grads

				# the convolution and guided gradients have a batch dimension
				# (which we don't need) so let's grab the volume itself and
				# discard the batch
				convOutput = convOutputs[0]
				guidedGrad = guidedGrads[0]
				
				# compute the average of the gradient values, and using them
				# as weights, compute the ponderation of the filters with
				# respect to the weights
				weights = tf.reduce_mean(guidedGrad, axis=(0, 1))
				cam = tf.reduce_sum(tf.multiply(weights, convOutput), axis=-1)
				
				# grab the spatial dimensions of the input image and resize
				# the output class activation map to match the input image
				# dimensions
				(w, h) = (np.shape(image)[1], np.shape(image)[0])
				heatmap = cv2.resize(cam.numpy(), (w, h))
				# normalize the heatmap such that all values lie in the range
				# [0, 1], scale the resulting values to the range [0, 255],
				# and then convert to an unsigned 8-bit integer
				numer = heatmap - np.min(heatmap)
				denom = (heatmap.max() - heatmap.min()) + eps
				heatmap = numer / denom
				heatmap = (heatmap * 255).astype("uint8")

				#im = np.array(image*255).astype('uint8')

				(heatmap, output) = self.overlay_heatmap(heatmap, image)


				cv2.imwrite(path + f"{i}/both.png", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
				cv2.imwrite(path + f"{i}/heatmap.png", cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
				cv2.imwrite(path + f"{i}/image.png",  cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		return
	
	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)
	

