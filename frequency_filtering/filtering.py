# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import numpy as np


class Filtering:

    def __init__(self, image):
        """initializes the variables for frequency filtering on an input image
        takes as input:
        image: the input image
        """
        self.image = image
        self.mask = self.get_mask

    def get_mask(self, shape):
        """Computes a user-defined mask
        takes as input:
        shape: the shape of the mask to be generated
        rtype: a 2d numpy array with size of shape
        """

        res = np.ones(shape)
        res[235:250, 220:247] = res[260:290, 265:290] = 0
        res[220:240, 280:320] = res[260:290, 180:220] = 0

        return res

    def post_process_image(self, image):
        """Post processing to display DFTs and IDFTs
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        You can perform post processing as needed. For example,
        1. You can perfrom log compression
        2. You can perfrom a full contrast stretch (fsimage)
        3. You can take negative (255 - fsimage)
        4. etc.
        """

        image = 20 * np.log10(0.1 + image)
        minimum = 255
        maximum = 0
        rows = image.shape[0]
        columns = image.shape[1]
        for i in range(rows):
            for j in range(columns):
                if image[i][j] < minimum:
                    minimum = image[i][j]
                if image[i][j] > maximum:
                    maximum = image[i][j]


        for i in range(rows):
            for j in range(columns):
                image[i][j] = ((image[i][j] - minimum) / (maximum - minimum))*255
        

        return image

    def filter(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of frequency_filtering, magnitude of filtered frequency_filtering
        ----------------------------------------------------------
        You are allowed to use inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do post processing on the magnitude and depending on the algorithm (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of frequency_filtering, magnitude of filtered frequency_filtering: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        """

        noisy_img = self.image
        noisy_img_fft = np.fft.fft2((noisy_img))

        noisy_img_fft_shifted_before = np.fft.fftshift(noisy_img_fft)
        noisy_img_fft_shifted = noisy_img_fft_shifted_before * (self.get_mask(noisy_img_fft_shifted_before.shape))
        im_out = np.abs(np.fft.ifft2(np.fft.ifftshift(noisy_img_fft_shifted)))
        noisy_img_fft_shifted_before_output = self.post_process_image(np.abs(noisy_img_fft_shifted_before))
        noisy_img_fft_shifted_output = self.post_process_image(np.abs(noisy_img_fft_shifted))
        
        return [im_out, noisy_img_fft_shifted_before_output, noisy_img_fft_shifted_output]
