import numpy as np

# -----------------------------------------------------------
# 1) TOY DATASET: TWO "IMAGES" (5x5), BINARY LABELS
# -----------------------------------------------------------
image1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=float)

image2 = np.array([
    [2, 2, 2, 2, 2],
    [2, 1, 1, 1, 2],
    [2, 1, 0, 1, 2],
    [2, 1, 1, 1, 2],
    [2, 2, 2, 2, 2]
], dtype=float)

images = np.stack([image1, image2], axis=0)  # shape: (2, 5, 5)
labels = np.array([0, 1])  # True labels for demonstration

print("Images shape:", images.shape)
print("Labels:", labels)

# -----------------------------------------------------------
# 2) DEFINE A SINGLE 3x3 KERNEL FOR CONV
# -----------------------------------------------------------
kernel = np.array([
    [1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]
], dtype=float)

print("\nKernel (3×3):\n", kernel)

# Manual convolution parameters
k_h, k_w = kernel.shape  # (3, 3)
stride = 1


# -----------------------------------------------------------
# 3) CONVOLVE EACH IMAGE WITH THE KERNEL
# -----------------------------------------------------------
def convolve_2d(image, kernel):
    """
    Manually convolve a 2D image with a 2D kernel (stride=1, no padding).
    Returns the feature map.
    """
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    out_h = i_h - k_h + 1
    out_w = i_w - k_w + 1

    feature_map = np.zeros((out_h, out_w), dtype=float)

    for y in range(out_h):
        for x in range(out_w):
            region = image[y:y + k_h, x:x + k_w]
            # Dot product
            feature_map[y, x] = np.sum(region * kernel)
    return feature_map


feature_maps = []
for i in range(images.shape[0]):
    fm = convolve_2d(images[i], kernel)
    feature_maps.append(fm)
    print(f"\n--- Image {i} ---")
    print("Raw Image:\n", images[i])
    print("Feature Map (3×3):\n", fm)

# Stack them => shape (2, 3, 3)
feature_maps = np.stack(feature_maps, axis=0)
print("\nAll feature maps shape:", feature_maps.shape)  # (2,3,3)

# -----------------------------------------------------------
# 4) FLATTEN AND PREPARE FOR DENSE LAYER
# -----------------------------------------------------------
batch_size, fm_h, fm_w = feature_maps.shape
flat_size = fm_h * fm_w
flattened = feature_maps.reshape(batch_size, flat_size)  # shape => (2,9)

for i in range(batch_size):
    print(f"\nFlattened vector for Image {i} (size {flat_size}):\n", flattened[i])

# -----------------------------------------------------------
# 5) DENSE LAYER (9 inputs -> 2 outputs)
# -----------------------------------------------------------
# We'll define random weights for demonstration
np.random.seed(42)
W = np.random.randn(flat_size, 2)  # shape (9,2)
b = np.random.randn(2)

print("\nDense layer weights W (9×2):\n", W)
print("Dense layer bias b (2,):\n", b)

# Forward pass through dense: logits = flattened × W + b
z = np.dot(flattened, W) + b  # shape => (2,2)

for i in range(batch_size):
    print(f"\nLogits for Image {i} (2 outputs): {z[i]}")


# -----------------------------------------------------------
# 6) SOFTMAX
# -----------------------------------------------------------
def softmax(logits):
    # logits shape: (batch_size, 2)
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


probs = softmax(z)  # shape => (2,2)

for i in range(batch_size):
    print(f"\nProbabilities for Image {i}: {probs[i]} (sum = {probs[i].sum()})")

# -----------------------------------------------------------
# 7) PREDICTED CLASS
# -----------------------------------------------------------
pred_classes = np.argmax(probs, axis=1)
for i in range(batch_size):
    print(f"\nImage {i} predicted class: {pred_classes[i]} | True label: {labels[i]}")
