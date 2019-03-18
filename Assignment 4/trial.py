class ResNet:
@staticmethod
def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
	shortcut = data

	# the first block of the ResNet module are the 1x1 CONVs
	bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
	momentum=bnMom)(data)
	act1 = Activation("relu")(bn1)

	conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

	bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
	act2 = Activation("relu")(bn2)
	conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

	bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
	act3 = Activation("relu")(bn3)
	conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

	if red:
		shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)