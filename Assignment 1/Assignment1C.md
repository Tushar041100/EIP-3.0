##### Name: Vishal V
##### Email: vishal114186@gmail.com
##### Batch: 3
--

# 1. Convolution
- Convolution is a mathematical function applied on an input image via a filter to extract specific information from it. The operation is similar to a spatial feature extraction applied on the different snippets of the input image. 
![ALT-IMG](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_1.png)
- In the figure, `f` is the input image and `g` is the filter and the convolution operation is the `integration` of the operation appled on all spatial snippets of the image. In the context of audio processing, convolution performs signal smoothening.
- The beauty of this operation lies in the fact that, the operation itself involves shared parameters and allows useful information gain if a suitable filter is used. Convolution also shows sparcity of connections as the output depends only on small number of parameters.
--- 
# 2. Filter/Kernel
- A filter/kernel is a feature extractor that is used in a convolution operation to extract specific information from the inputs. The kernel weights could be trained using backpropagation or SoTA filters like the `Sobel filter` or `Scharr filter` could be used. 
- A filter is anlogous to localized aggregation in a neural network. It is a weighted kernel (centered) that preserves different weighted pixels for more effective feature extraction. .
- filter/kernels that are first randomly initialized and then updated using backpropagation/genetic algorithms capture the stats of the data much better than other pre-built filters. The specificity of the filter's information gain deepens as the image becomes more niche as the conv layers increase
![ALT_IMG](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_8.png)
![ALT-IMG](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_6.png)

---
# 3. 3x3 Convolution
- A 3x3 convolution operation is one that is performed using a 3x3 filter/kernel and has 9 values chosen by backpropagation or strategic research. 
- A 3x3 convolution is the most useful filter size because it is the smallest filter that can extract a feature and is also used in conjunction with NVIDIA CUDA drivers for accelerated processing.
- It is used for identity, blurring, unsharp masking, sharpening, embossing, edge detection(vertical edge/horizontal edge) etc
![ALT-IMG](https://www.codingame.com/servlet/fileservlet?id=23779693390196)
---
# 4. Activation Functions
- Activation functions are non-linear statistical functions applied on outputs of a perceptron or a layer in a neural network to maximise information gain and improve the learning rate. The output from an activation function is fed as input into the next layer.
- There are several types of activation functions, each intended for a specific purpose for example, the logistic activation function maps in the 0-1 range while the ReLU activation function is defined as f(x) = max(0, x)
- Using a linear activation or none would result in the weights and biases of the network to simply learn a linear mapping/transformation of the data. This regression model cannot be applied to complex problem domains such as sequence models or image processing.
![ALT-IMG](https://cdn-images-1.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png)
---
# 5. Receptive Field
- Receptive field of a CNN is the region in the input that the filter is taking into consideration at that instant. A receptive field of a feature can be described by its epicentre (central location) and its size. 
- It must be noted that all pixels do not necessarily have useful information to learn corresponding to a feature extractor. Within a receptive field, the closer a pixel is to the center of the field, the more it contributes to the calculation of the output feature. 
![ALT-IMG](https://cdn-images-1.medium.com/max/720/1*B56Ibp2x4BXSwhLkcbq1SA.png)
- One way to overcome this is to use a larger stride or pad the input image with zeros so that all pixels are approximately considered equally. The following formula is used to calculate the effective dimensions of the output after a convolution operation (with a filter is applied on) the input image
![ALT-IMG](https://cdn-images-1.medium.com/max/720/1*3V6TJG1U0uEPp8VUxjORpQ.png)

---
