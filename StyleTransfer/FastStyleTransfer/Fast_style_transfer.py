import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# NNIDIA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("GPU available: %s\n" % (tf.config.list_physical_devices('GPU')))

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image


def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""

    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


content_image_path = 'dog.jpg'
style_image_path = 'style2.jpg'
output_image_size = 1000

# The content image size can be arbitrary.
content_img_size = (output_image_size, output_image_size)
# The style prediction model was trained with image size 256 and it's the 
# recommended image size for the style image (though, other sizes work as 
# well but will lead to different results).
style_img_size = (256, 256)  # Recommended to keep it at 256.

content_image = load_image(content_image_path, content_img_size)
style_image = load_image(style_image_path, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

# Load TF-Hub module.
hub_handle = 'model'
hub_module = hub.load(hub_handle)

# Stylize content image with given style image.
# This is pretty fast within a few milliseconds on a GPU.
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0].numpy().reshape((output_image_size, output_image_size, 3))
print(type(stylized_image))
img = 255 * stylized_image # Now scale by 255
img = img.astype(np.uint8)
im = Image.fromarray(img, "RGB")
im.save("Stylized_image.jpg")
print("Saved as Stylized_image.jpg")


