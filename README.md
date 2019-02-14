# Image Compression with Deterministic K-Means

This project consists of the implementation of the k-means algorithm with deterministic initialization and an utility to "compress" images using this algorithm.

Even though the k-means algorithms is a programmed as a stand-alone function, it is optimized to have better performance when used for compressing images.

Here is a sample of what this implementation can do:

## Original Image
<img src="samples/marilu3_original.png" alt="Uncompressed image" width="550"/>

## Compressed Images:
### 2 Colors
<img src="samples/marilu3_2colors.png" alt="Uncompressed image" width="550"/>
### 4 Colors
<img src="samples/marilu3_4colors.png" alt="Uncompressed image" width="550"/>
### 8 Colors
<img src="samples/marilu3_8colors.png" alt="Uncompressed image" width="550"/>
### 16 Colors
<img src="samples/marilu3_16colors.png" alt="Uncompressed image" width="550"/>
### 32 Colors
<img src="samples/marilu3_32colors.png" alt="Uncompressed image" width="550"/>

The objective of this implementation was finding different deterministic
initialization functions to compare them and see which obtains a better MSE. This is still under development since the implementation is still a work in process
but I'm getting there.
