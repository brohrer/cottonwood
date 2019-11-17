import numpy as np


def indent(unindented, n_spaces=2):
    """
    Indent a multi-line string using spaces.
    """
    indent = " " * n_spaces
    newline = "\n" + indent
    return indent + newline.join(unindented.split("\n"))


def gamma_decompress(img):
    """
    Make pixel values perceptually linear.
    """
    img_lin = ((img + 0.055) / 1.055) ** 2.4
    i_low = np.where(img <= .04045)
    img_lin[i_low] = img_lin[i_low] / 12.92
    return img_lin


def gamma_compress(img_lin):
    """
    Make pixel values display-ready.
    """
    img = 1.055 * img_lin ** (1 / 2.4) - 0.055
    i_low = np.where(img_lin <= .0031308)
    img[i_low] = 12.92 * img_lin[i_low]
    return img


def rgb2gray_linear(rgb_img):
    """
    Convert *linear* RGB values to *linear* grayscale values.
    """
    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]

    gray_img = (
        0.2126 * red
        + 0.7152 * green
        + 0.0722 * blue)

    return gray_img


def rgb2gray_approx(rgb_img):
    """
    Convert *linear* RGB values to *linear* grayscale values.
    """
    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]

    gray_img = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue)

    return gray_img


def rgb2gray(rgb_img):
    """
    rgb_img is a 3-dimensional Numpy array of type float with
    values ranging between 0 and 1.
    Dimension 0 represents image rows, left to right.
    Dimension 1 represents image columns top to bottom.
    Dimension 2 has a size of 3 and
    represents color channels, red, green, and blue.

    For more on what this does and why:
    https://brohrer.github.io/convert_rgb_to_grayscale.html

    Returns a gray_img 2-dimensional Numpy array of type float.
    Values range between 0 and 1.
    """
    return gamma_compress(rgb2gray_linear(gamma_decompress(rgb_img)))


def difference_img(img1, img2):
    """
    Create a difference image, indicating both the magnitude and
    the direction of the difference between two images,
    pixel by pixel.
    """
    scale = 4
    img_diff_lin = scale * (gamma_decompress(img1) - gamma_decompress(img2))
    i_neg = np.where(img_diff_lin < 0)
    # Temporarily convert negative values to positive to avoid breaking
    # the gamma compression method.
    img_diff_lin[i_neg] *= -1
    img_diff = gamma_compress(img_diff_lin)
    img_diff[i_neg] *= -1
    return img_diff
