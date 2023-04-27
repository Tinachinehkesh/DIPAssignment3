import numpy as np


class Filtering:

    def __init__(self, image):
        self.image = image

    def apply(self, image, kernel):  
        kernel_row, kernel_column = kernel.shape
        image_row, image_column = image.shape
        res = np.zeros(image.shape)

        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_column - 1) / 2)

        img_with_pad = np.zeros((image_row + (2 * pad_height), image_column + (2 * pad_width)))
        img_with_pad[pad_height:img_with_pad.shape[0] - pad_height, pad_width:img_with_pad.shape[1] - pad_width] = image

        for row in range(image_row):
            for column in range(image_column):
                res[row, column] = np.sum(kernel * img_with_pad[row:row + kernel_row, column:column + kernel_column])
                res[row, column] /= kernel.shape[0] * kernel.shape[1]

        return res
    
    def generate_gaussian_kernel(self, size, sigma):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power((kernel_1D[i]) / sigma, 2) / 2)

        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()

        return kernel_2D

    def get_gaussian_filter(self):
        """Initialzes/Computes and returns a 5X5 Gaussian filter"""

        gaussian_kernel = self.generate_gaussian_kernel(5, 2)
        return self.apply(self.image, gaussian_kernel)

    def get_laplacian_filter(self):
        """Initialzes and returns a 3X3 Laplacian filter"""

        laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        res = self.apply(self.image, laplacian_kernel)
        return res

    def filter(self, filter_name):
        """Perform filtering on the image using the specified filter, and returns a filtered image
            takes as input:
            filter_name: a string, specifying the type of filter to use ["gaussian", laplacian"]
            return type: a 2d numpy array
                """

        if filter_name == "gaussian":
            return self.get_gaussian_filter()
        elif filter_name == "laplacian":
            return self.get_laplacian_filter()