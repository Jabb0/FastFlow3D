import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# https://stackoverflow.com/questions/52540037/create-image-using-matplotlib-imshow-meshgrid-and-custom-colors

def flow_to_rgb(flows):
    """
    Convert a flow to a rgb value
    Args:
        flows: (N, 3) vector flow

    Returns: (N, 3) RGB values normalized between 0 and 1

    """
    # https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((flows.shape[0], 1, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flows[..., 0], flows[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    rgb = rgb[:, 0, :] / 255.  # Normalize to 1

    rgb[rgb < 0.2] = 0.2  # Just for visualize not moving points

    return rgb



def colour_wheel(samples=1024):
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))

    flows = np.vstack([yy.ravel(), xx.ravel()]).T
    rgb_flows = flow_to_rgb(flows)
    rgb_flows = np.reshape(rgb_flows, (samples, samples, 3))

    res = np.zeros((xx.shape[0], xx.shape[1], 3))
    for i in range(xx.shape[0]):
        print(i)
        for j in range(xx.shape[1]):
            res[i, j, :] = rgb_flows[i, j]

    plt.figure(dpi=100)
    plt.imshow(res)
    plt.show()

colour_wheel()