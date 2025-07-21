import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2
from PIL import Image


def classify_irregularity(score):
    if score < 0.1:
        return "Smooth"
    elif score < 0.3:
        return "Moderate"
    else:
        return "Highly Irregular"
# Updated version: exclude greasy spots outside the leaf by masking non-leaf areas


def get_leaf_mask(image):
    """Return binary mask of the leaf region based on HSV color range."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (15, 40, 40), (90, 255, 255))


def detect_irregular_greasy_spots_with_leaf_mask(image):
    """Detects greasy spots only inside the leaf area and evaluates shape irregularity."""
    leaf_mask = get_leaf_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply leaf mask to grayscale image before thresholding
    masked_gray = cv2.bitwise_and(gray, gray, mask=leaf_mask)

    thresh = cv2.adaptiveThreshold(
        masked_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=3
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    greasy_spots = []
    irregularities = []
    result = image.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if 20 < area < 500:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            irregularity = 1 - circularity
            irregularities.append(irregularity)
            greasy_spots.append(c)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)

    avg_irreg = np.mean(irregularities) if irregularities else 0
    irreg_label = classify_irregularity(avg_irreg)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result_rgb, len(greasy_spots), avg_irreg, irreg_label


image_path = 'F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\images\\h\\tomato-leaf.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))

# Run updated method
updated_img, spot_count, avg_irregularity, category = detect_irregular_greasy_spots_with_leaf_mask(
    image)

# Show updated image
plt.figure(figsize=(6, 6))
plt.imshow(updated_img)
plt.title(
    f"Spots: {spot_count}, Irregularity: {avg_irregularity:.2f} ({category})")
plt.axis("off")
plt.tight_layout()
plt.show()
