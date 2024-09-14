import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to load and preprocess the image
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    # img = img.astype('float32') / 255.0  # Normalize the image
    return img


# Function to add Gaussian noise to the image
def add_noise(image):
    noise = np.random.normal(loc=0, scale=0.1, size=image.shape)
    noisy_image = image + noise
    # noisy_image = np.clip(noisy_image, 0.0, 1.0)
    return noisy_image


# Function to apply a blur filter to simulate blur interpretation
def apply_blur(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image


# Vision Transformer model for denoising
class VisionTransformer(layers.Layer):
    def __init__(self, num_patches, projection_dim, num_heads, transformer_units):
        super(VisionTransformer, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.transformer_units = transformer_units

    def call_out(self, inputs):
        patches = self._extract_patches(inputs)
        encoded_patches = self._encode_patches(patches)
        return self._transform(encoded_patches)

    def _extract_patches(self, inputs):
        # Divide image into patches
        patch_size = 16
        patches = tf.image.extract_patches(images=inputs,
                                           sizes=[1, patch_size, patch_size, 1],
                                           strides=[1, patch_size, patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patches = tf.reshape(patches, [-1, self.num_patches, patch_size * patch_size])
        return patches

    def _encode_patches(self, patches):
        # Linearly project patches
        encoded = self.projection(patches)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = self.position_embedding(positions)
        return encoded + position_embeddings

    def _transform(self, encoded_patches):
        x = self.attention_layer(encoded_patches, encoded_patches)
        x = tf.keras.layers.LayerNormalization()(x)
        for units in self.transformer_units:
            x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        return x


def build_denoising_transformer_model():
    inputs = layers.Input(shape=(128, 128, 1))

    # Apply initial blur interpretation
    blurred = layers.Lambda(lambda x: tf.image.adjust_contrast(x, 0.5))(inputs)

    # Use Vision Transformer for denoising
    transformer_layer = VisionTransformer(num_patches=64, projection_dim=64, num_heads=4, transformer_units=[128, 64])
    denoised_patches = transformer_layer.call_out(blurred)

    # Reconstruction layer to generate output image
    recon_img = layers.Dense(16 * 16, activation='sigmoid')(denoised_patches)
    recon_img = layers.Reshape((128, 128, 1))(recon_img)

    model = models.Model(inputs, recon_img)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Display images
def display_images(original, noisy, denoised):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.squeeze(original), cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(np.squeeze(noisy), cmap='gray')
    ax[1].set_title("Noisy Image")
    ax[2].imshow(np.squeeze(denoised), cmap='gray')
    ax[2].set_title("Denoised Image")
    plt.show()


# Load, add noise, and apply blur to the image
image_path = 'pexels-rdne-7468260.jpg'
# cvrt to grey , resize to 128, 128, normalize
original_image = load_image(image_path)

# random noise added to the image and set range with np clip 0-1
noisy_image = add_noise(original_image)

# apply Gaussian blur
blurred_noisy_image = apply_blur(noisy_image)

# Expand dims for batch and channels
original_image = np.expand_dims(original_image, axis=(0, -1))
blurred_noisy_image = np.expand_dims(blurred_noisy_image, axis=(0, -1))

# Build and train the model
denoising_transformer_model = build_denoising_transformer_model()
denoising_transformer_model.fit(blurred_noisy_image, original_image, epochs=100, verbose=1)

# Denoise the image
denoised_image = denoising_transformer_model.predict(blurred_noisy_image)

# Display the images
display_images(original_image, blurred_noisy_image, denoised_image)



#   ---------------------- hybrid sift feature matching ----------
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# Match descriptors
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# Ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3)
plt.show()