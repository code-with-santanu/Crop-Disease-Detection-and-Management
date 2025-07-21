# Re-import libraries after environment reset
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reload
imagepath = 'F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\images\\h\\istockphoto-1266605044-612x612.jpg'
image = cv2.imread(imagepath)
image = cv2.resize(image, (256, 256))

# Define function


def visualize_hsv_shift(image):
    min_lesion_pixels = 100
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lesion and healthy masks
    lesion_mask = cv2.inRange(hsv, (5, 100, 50), (25, 255, 200))
    healthy_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

    lesion_pixel_count = np.count_nonzero(lesion_mask)

    if lesion_pixel_count > min_lesion_pixels:
        mean_lesion = np.array(cv2.mean(hsv, mask=lesion_mask)[:3])
        mean_healthy = np.array(cv2.mean(hsv, mask=healthy_mask)[:3])
        hsv_shift = np.linalg.norm(mean_lesion - mean_healthy)
    else:
        hsv_shift = 0  # Too few pixels to trust measurement

    # Annotate image
    visual = image.copy()
    visual[lesion_mask > 0] = [0, 0, 255]   # red
    visual[healthy_mask > 0] = [0, 255, 0]  # green
    visual_rgb = cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)

    return visual_rgb, round(hsv_shift, 2), lesion_pixel_count


# Apply analysis
highlighted_img, hsv_shift_val, count = visualize_hsv_shift(image)

# Show result
plt.figure(figsize=(6, 6))
plt.imshow(highlighted_img)
plt.title(f"HSV Shift (Lesion → Healthy): {hsv_shift_val}")
plt.axis("off")
plt.tight_layout()
plt.show()

hsv_shift_val
