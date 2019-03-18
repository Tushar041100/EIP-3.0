from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.convolutional import AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation
from clr_callback import *
from keras.layers import Flatten, Input, add
from keras.optimizers import Adam
from keras.callbacks import *
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K

class ResNet:
	@staticmethod
	def bottleneck_residual_model(data, kernels, strides, chanDim, reduced=False, reg=0.0001, epsilon=2e-5, mom=0.9):
		# chanDim specifies the axis to apply batch-norm on
		shortcut = data
		
		bn1 = BatchNormalization(axis=chanDim, epsilon=epsilon, momentum=mom)(data)
		act1 = Activation("relu")(bn1)
		conv1 = Conv2D(int(kernels * 0.25), (1,1), strides=strides, use_bias=False, kernel_regularizer=l2(reg))(act1)

		bn2 = BatchNormalization(axis=chanDim, epsilon=epsilon, momentum=mom)(conv1)
		act2 = Activation("relu")(bn2)
		conv2 = SeparableConv2D(int(kernels * 0.25), (3,3), padding='same', strides=strides, use_bias=False, depthwise_regularizer=l2(reg))(act2)

		bn3 = BatchNormalization(axis=chanDim, epsilon=epsilon, momentum=mom)(conv2)
		act3 = Activation("relu")(bn3)
		conv3 = Conv2D(kernels, (1,1), strides=strides, use_bias=False, kernel_regularizer=l2(reg))(act3)

		if True:
			input_shape = K.int_shape(data)
			residual_shape = K.int_shape(conv3)
			stride_width = int(round(input_shape[1] / residual_shape[1]))
			stride_height = int(round(input_shape[2] / residual_shape[2]))
			equal_channels = input_shape[3] == residual_shape[3]

			shortcut = act1
			# 1 X 1 conv if shape is different. Else identity.
			if stride_width > 1 or stride_height > 1 or not equal_channels:
				shortcut = Conv2D(filters=residual_shape[3],
	                      kernel_size=(1, 1),
	                      strides=(stride_width, stride_height),
	                      padding="valid",
	                      kernel_initializer="he_normal",
	                      kernel_regularizer=l2(0.0001))(act1)
				# shortcut = Conv2D(kernels, (1, 1), strides=(stride_width, stride_height), use_bias=False, kernel_regularizer=l2(reg))(act1)

		x = add([conv3, shortcut])

		return x

	@staticmethod
	def build(width, height, depth, classes, stages, filters, reg=0.0001, epsilon=2e-5, mom=0.9):

		inputShape = (height, width, depth)
		inputs = Input(shape=inputShape)
		chanDim= -1

		x = BatchNormalization(axis=chanDim, epsilon=epsilon, momentum=mom)(inputs)
		x = SeparableConv2D(filters[0], (5, 5), use_bias=False, padding="same", depthwise_regularizer=l2(reg))(x)
		x = BatchNormalization(axis=chanDim, epsilon=epsilon, momentum=mom)(x)
		x = Activation("relu")(x)
		x = ZeroPadding2D((1, 1))(x)
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)
		
		for i in range(0, len(stages)):

			if i==0:
				strides = (1,1)
			else:
				strides = (2,2)
			x = ResNet.bottleneck_residual_model(x, filters[i+1], strides, chanDim, reduced=True)

			for j in range(0, stages[i] - 1):
				x = ResNet.bottleneck_residual_model(x, filters[i+1], (1,1), chanDim, epsilon=epsilon, momentum=mom)


		x = BatchNormalization(axis=chanDim, epsilon=epsilon, momentum=mom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((8, 8))(x)


		x = Flatten()(x)
		x = Activation("softmax")(x)

		model = Model(inputs, name="resnet")
		return model

