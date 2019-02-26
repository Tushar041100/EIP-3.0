##### Name: Vishal V
##### Email: vishal114186@gmail.com
##### Batch: 3
--

# 1. Convolution
- Convolution is a mathematical function appplied on an input image via a filter to extract specific information from it. The operation is similar to a spatial feature extraction applied on the different snippets of the input image. 
![ALT-IMG](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/70_blog_image_1.png)
- In the figure, `f` is the input image and `g` is the filter and the convolution operation is the `integration` of the operation appled on all spatial snippets of the image.In the context of audio processing, convolution performs signal smoothening.
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
---
# 4. Activation Functions
---
# 5. Receptive Field
---
