# Neural Style Transfer & Neural Doodles

Neural Style Transfer (NST) is a fascinating technique in deep learning that combines the content of one image with the style of another, producing a new image that maintains the original content but adopts the artistic style of the second image. This method was introduced by Gatys et al. in their seminal paper, "A Neural Algorithm of Artistic Style" .

# Implementing Neural Style Transfer
There are several ways to implement NST, with popular frameworks being PyTorch and TensorFlow.

Using PyTorch
PyTorch provides a comprehensive tutorial on implementing NST, which includes:

-> Loading and preprocessing images.

-> Defining the model and loss functions.

-> Running the style transfer optimization
# Examples
## Single Style Transfer
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/content/blue-moon-lake.jpg" width=49% height=300 alt="blue moon lake"> <img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" width=49% height=300 alt="starry night">
<br><br> Results after 100 iterations using the INetwork<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Blue-Moon-Lake_at_iteration_100.jpg?raw=true" width=98% height=450 alt="blue moon lake style transfer">
<br><br> DeepArt.io result (1000 iterations and using improvements such as Markov Random Field Regularization) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/DeepArt_Blue_Moon_Lake.jpg" width=98% height=450>

## Style Interpolation
Style weight and Content weight can be manipulated to get drastically different results.

Leonid Afremov's "Misty Mood" (Original Source: https://afremov.com/) is the style image and "Dipping Sun" is the content image : <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Dipping-Sun.jpg?raw=true" height=300 width=49%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/misty-mood-leonid-afremov.jpg?raw=true" height=300 width=50%> 

<table>
<tr align='center'>
<td>Style=1, Content=1000</td>
<td>Style=1, Content=1</td>
<td>Style=1000, Content=1</td>
</tr>
<tr>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/DippingSun3.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/DippingSun2.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/DippingSun1.jpg?raw=true" height=300></td>
</tr>
</table>


## Multiple Style Transfer
The next few images use the Blue Moon Lake as a content image and Vincent Van Gogh's "Starry Night" and Georgia O'Keeffe's "Red Canna" as the style images: <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" width=49% height=300> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/red-canna.jpg?raw=true" height=300 width=49%>

The below are the results after 50 iterations using 3 different style weights : <br>
<table align='center'>
<tr align='center'>
<td>Starry Night : 1.0, Red Canna 0.2</td>
<td>Starry Night : 1.0, Red Canna 0.4</td>
<td>Starry Night : 1.0, Red Canna 1.0</td>
</tr>
<tr>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/blue_moon_lake_1-0_2.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/blue_moon_lake_1-0_4.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/blue_moon_lake_1-1_at_iteration_50.jpg?raw=true" height=300></td>
</tr>
</table>

## All Transfer Techniques
Each of these techniques can be used together, or in stages to generate stunning images. 

In the folowing image, I have used Masked style transfer in a multi scale style transfer technique - with scales of 192x192, 384x384, 768x768, applied a super resolution algorithm (4x and then downscaled to 1920x1080), applied color transfer and mask transfer again to sharpen the edges, used a simple sharpening algorithm and then finally denoise algorithm.

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/ancient_city.jpg?raw=true" width=33% alt="ancient city japanese" height=250> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/blue_swirls.jpg?raw=true" width=33% height=250> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/ancient-city.jpg?raw=true" width=33% height=250> 

Result : <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/ancient_city_multiscale.jpg?raw=true" width=99% alt="ancient city japanese">


# Neural Doodle Examples
Renoit Style + Content Image <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/neural_doodle/generated/renoit_new.png?raw=true" width=98%><br>
Monet Style + Doodle Creation <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/neural_doodle/generated/monet_new.png?raw=true" width=98%>
<br>Van Gogh + Doodle Creation <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/neural_doodle/generated/van%20gogh.png?raw=true" width=98%>

## Weights (VGG 16)

Weights are now automatically downloaded and cached in the ~/.keras (Users/<username>/.keras for Windows) folder under the 'models' subdirectory. The weights are a smaller version which include only the Convolutional layers without Zero Padding Layers, thereby increasing the speed of execution.

Note: Requires the latest version of Keras (1.0.7+) due to use of new methods to get files and cache them into .keras directory.

## Modifications to original implementation :
- Uses 'conv5_2' output to measure content loss.
Original paper utilizes 'conv4_2' output

- Initial image used for image is the base image (instead of random noise image)
This method tends to create better output images, however parameters have to be well tuned.
Therefore their is a argument 'init_image' which can take the options 'content' or 'noise'

- Can use AveragePooling2D inplace of MaxPooling2D layers
The original paper uses AveragePooling for better results, but this can be changed to use MaxPooling2D layers via the argument `--pool_type="max"`. By default MaxPooling is used, since if offers sharper images, but AveragePooling applies the style better in some cases (especially when style image is the "Starry Night" by Van Gogh).

- Style weight scaling
- Rescaling of image to original dimensions, using lossy upscaling present
- Maintain aspect ratio of intermediate and final stage images, using lossy upscaling

## Improvements in INetwork
- Improvement 3.1 in paper : Geometric Layer weight adjustment for Style inference
- Improvement 3.2 in paper : Using all layers of VGG-16 for style inference
- Improvement 3.3 in paper : Activation Shift of gram matrix
- Improvement 3.5 in paper : Correlation Chain

These improvements are almost same as the Chain Blurred version, however a few differences exist : 
- Blurring of gram matrix G is not used, as in the paper the author concludes that the results are often not major, and convergence speed is greatly diminished due to very complex gradients.
- Only one layer for Content inference instead of using all the layers as suggested in the Chain Blurred version.
- Does not use CNN MRF network, but applies these modifications to the original algorithm.
- All of this is applied on the VGG-16 network, not on the VGG-19 network. It is trivial to extrapolate this to the VGG-19 network. Simply adding the layer names to the `feature_layers` list will be sufficient to apply these changes to the VGG-19 network. 


### Benefits 
- Allows Style Transfer, Neural Doodles, Color Transfer and Masked Style Transfer easily
- Automatically executes the script based on the arguments.
- Easy selection of images (Content, Style (Multiple Selection allowed), Output Prefix)
- Easy parameter selection
- Easily generate argument list, if command line execution is preferred. 
- Creates log folders for each execution so settings can be preserved
- Runs on Windows (Native) and Linux (Using Mono)

To use multiple style images, when the image choice window opens, select all style images as needed. Pass multiple style weights by using a space between each style weight in the parameters section.


# Network.py in action
![Alt Text](https://raw.githubusercontent.com/titu1994/Neural-Style-Transfer/master/images/Blue%20Moon%20Lake.gif)

# Requirements 
- Theano / Tensorflow
- Keras 
- CUDA (GPU) -- Recommended
- CUDNN (GPU) -- Recommended
- Numpy
- h5py
- Scipy + PIL + Scikit-image

# Speed
On a 980M GPU, the time required for each epoch depends on mainly image size (gram matrix size) :

For a 400x400 gram matrix, each epoch takes approximately 8-10 seconds. <br>
For a 512x512 gram matrix, each epoch takes approximately 15-18 seconds. <br>
For a 600x600 gram matrix, each epoch takes approximately 24-28 seconds. <br>

For Masked Style Transfer, the speed is now same as if using no mask. This was acheived by preventing gradient computation of the mask multiplied with the style and content features.

For Multiple Style Transfer, INetwork.py requires slightly more time (~2x single style transfer as shown above for 2 styles, ~3x for 3 styles and so on). Results are better with INetwork.py in multiple style transfer.

For Multi Style Multi Mask Style Transfer, the speed is now same as if using multiple styles only. It was acheived by preventing gradient computation of the mask multiplied with the style and content features.

- For multi style multi mask network, Network.py requires roughly 24 (previously 72) seconds per iteration, whereas INetwork.py requires 87 (previously 248) seconds per iteration
  


