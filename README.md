# Image Compression with Deterministic K-Means

This project consists of the implementation of the k-means algorithm 
with deterministic initialization and an utility to "compress" images
using this algorithm.

Even though the k-means algorithms is a programmed as a stand-alone function,
it is optimized to have better performance when used for compressing images.

Here is a sample of what this implementation can do:

![uncompressed image](samples/marilu3_original.png?raw=true "Uncompressed image")
![uncompressed image](samples/marilu3_2colors.png?raw=true "Compressed image with 2 colors")
![uncompressed image](samples/marilu3_4colors.png?raw=true "Compressed image with 4 colors")
![uncompressed image](samples/marilu3_8colors.png?raw=true "Compressed image with 8 colors")
![uncompressed image](samples/marilu3_16colors.png?raw=true "Compressed image with 16 colors")
![uncompressed image](samples/marilu3_32colors.png?raw=true "Compressed image with 32 colors")

The objective of this implementation was finding different deterministic
initialization functions to compare them and see which obtains a better MSE.
This is still under development since the implementation is still a work in process
but I'm getting there.
