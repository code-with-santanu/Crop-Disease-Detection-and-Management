import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2
from PIL import Image


class ImageProcessor:
    def __init__(self, image_path):
        """
        Initialize the ImageAnalysis class with the path to an image.
        """
        self.results = {}

        self.image_path = image_path
        self.image = None
        self.gray = None
        self.hsv = None
        self.image_rgb = None

    ########### Initial Processing Functions ###########

    def load_image(self):
        """
        Load the image from the given path.
        """
        try:
            # self.image = Image.open(self.image_path)
            self.image = cv2.imread(self.image_path)
            print("Image loaded successfully.")

            # Resize the image to a standard size (256x256)
            self.image = cv2.resize(self.image, (256, 256))

            # Convert to RGB for PIL compatibility
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Convert to grayscale and HSV
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        except Exception as e:
            print(f"Error loading image: {e}")

    def display_image(self):
        """
        Display the loaded image.
        """
        if self.image:
            self.image.show()
        else:
            print("No image loaded. Please load an image first.")

    def save_image(self, save_path):
        """
        Save the current image to the specified path.
        """
        if self.image:
            try:
                self.image.save(save_path)
                print(f"Image saved to {save_path}.")
            except Exception as e:
                print(f"Error saving image: {e}")
        else:
            print("No image loaded. Please load an image first.")

    def get_result(self):
        return self.results

    ########### Analysis Functions ###########

    # 1. brown-black Lesion Count
    def analyze_lesion_count(self):
        """
        Count the number of brown-black lesions in the image.
        """

        # Extract the lower 40% of the image (approximate "lower part of the leaf")
        height = self.image.shape[0]
        lower_part = self.image[int(height * 0.6):, :]  # Lower 40% region

        # Convert to HSV for color-based masking of brown/black lesions
        hsv_lower = cv2.cvtColor(lower_part, cv2.COLOR_BGR2HSV)

        # Define HSV color range for brownish-black lesions (adjust as needed)
        lower_brown = np.array([5, 50, 20])
        upper_brown = np.array([25, 255, 120])

        # Create a binary mask for brown lesions in the lower part
        brown_mask = cv2.inRange(hsv_lower, lower_brown, upper_brown)

        # Find contours of lesions in the masked area
        contours, _ = cv2.findContours(
            brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Measure lesion areas (filter out small ones)
        lesion_areas = [cv2.contourArea(c)
                        for c in contours if cv2.contourArea(c) > 30]
        avg_size = np.mean(lesion_areas) if lesion_areas else 0
        max_size = np.max(lesion_areas) if lesion_areas else 0

        self.results['brown_lesion_count'] = len(lesion_areas)

        return lesion_areas

    # 2. Average Spot Diameter
    def analyze_avg_spot_diameter(self):
        """
        Calculate the average diameter of spots.
        """
        # get the lesion areas from the previous function
        lesion_areas = self.analyze_lesion_count()
        diameter_pixels = [np.sqrt(4 * a / np.pi) for a in lesion_areas]
        avg_diameter_pixel = np.mean(diameter_pixels) if diameter_pixels else 0

        # convert avg diameter pixel to inch
        image_width_in_pixels = 256
        real_leaf_width_in_inches = 6
        pixels_per_inch = image_width_in_pixels / real_leaf_width_in_inches
        diameter_inch = avg_diameter_pixel / pixels_per_inch

        self.results['lesion_diameter'] = round(diameter_inch, 2)

    # 3. Ring Count using HoughCircles
    def analyze_ring_count(self):
        """
        Count the number of rings using HoughCircles.
        """
        # Step 1: Create a lesion mask (darker areas likely to be lesions)
        lesion_mask = cv2.inRange(self.gray, 0, 90)
        lesion_only = cv2.bitwise_and(self.gray, self.gray, mask=lesion_mask)

        # Step 2: Extract contours of lesion regions
        contours, _ = cv2.findContours(
            lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ring_count = 0
        annotated = self.image.copy()

        # Step 3: For each lesion, crop and apply HoughCircles locally
        for c in contours:
            if cv2.contourArea(c) < 100:  # Skip small regions (noise)
                continue

            x, y, w, h = cv2.boundingRect(c)
            roi = lesion_only[y:y+h, x:x+w]

            # Apply HoughCircles in ROI
            roi_blur = cv2.GaussianBlur(roi, (7, 7), 2)
            circles = cv2.HoughCircles(roi_blur, cv2.HOUGH_GRADIENT, 1.2, 10,
                                       param1=50, param2=18,
                                       minRadius=5, maxRadius=30)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    cv2.circle(annotated, (x + cx, y + cy), r, (0, 255, 0), 1)
                    cv2.circle(annotated, (x + cx, y + cy), 2, (0, 0, 255), -1)
                ring_count += len(circles)

        self.results['ring_count'] = ring_count

    # 4. Yellowing Area (%)
    def analyze_yellowing_area(self):
        """
        Calculate the percentage of yellowing area.
        """
        yellow_mask = cv2.inRange(self.hsv, (20, 80, 80), (35, 255, 255))

        # Detect full leaf area (green/brown tones)
        leaf_mask = cv2.inRange(self.hsv, (15, 40, 40), (90, 255, 255))
        # Keep only yellow pixels inside leaf
        yellow_on_leaf = cv2.bitwise_and(
            yellow_mask, yellow_mask, mask=leaf_mask)

        # Calculate areas
        leaf_area = np.sum(leaf_mask > 0)
        yellow_area = np.sum(yellow_on_leaf > 0)
        yellow_percent = (yellow_area / leaf_area) * \
            100 if leaf_area != 0 else 0

        self.results['yellow_area_percent'] = round(yellow_percent, 2)

    # 5. Greasy Lesion Shape Irregularity
    def analyze_shape_irregularity(self):
        """
        Detects greasy spots on a leaf and measures their count and shape irregularity.

        Returns:
            - Annotated image with bounding boxes
            - Count of greasy spots
            - Average shape irregularity (1 - circularity)
        """
        # find leaf mask to eliminate background
        leaf_mask = self.detect_leaf_mask()
        # Convert to grayscale
        gray = self.gray

        # Apply leaf mask to grayscale image before thresholding
        masked_gray = cv2.bitwise_and(gray, gray, mask=leaf_mask)

        # Apply adaptive threshold to highlight dark spots
        thresh = cv2.adaptiveThreshold(
            masked_gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=3
        )

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        greasy_spots = []  # store the greasy spots
        irregularities = []

        result = self.image.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if 20 < area < 500:  # Filter small noise and large blotches
                perimeter = cv2.arcLength(c, True)
                circularity = 4 * np.pi * area / \
                    (perimeter ** 2) if perimeter != 0 else 0
                irregularity = 1 - circularity  # Higher = more irregular
                irregularities.append(irregularity)
                greasy_spots.append(c)

                # Draw bounding box
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Compute average irregularity
        avg_irregularity = np.mean(irregularities) if irregularities else 0

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        self.results['gressy_lesion_count'] = len(greasy_spots)
        self.results['avg_lesion_irregularity'] = round(avg_irregularity, 2)

    # 6. HSV Color Shift
    def analyze_hsv_shift(self):
        """
        Calculates the HSV color shift between infected (brown/yellow) and healthy (green) regions.
        Useful for detecting color progression like 'yellow-green → brown-red'.
        """
        min_lesion_pixels = 100

        lesion_mask = cv2.inRange(self.hsv, (5, 100, 50), (25, 255, 200))
        healthy_mask = cv2.inRange(self.hsv, (35, 40, 40), (85, 255, 255))

        lesion_pixel_count = np.count_nonzero(lesion_mask)

        if lesion_pixel_count > min_lesion_pixels:
            mean_lesion = np.array(cv2.mean(self.hsv, mask=lesion_mask)[:3])
            mean_healthy = np.array(cv2.mean(self.hsv, mask=healthy_mask)[:3])
            self.results['hsv_shift'] = round(np.linalg.norm(
                mean_lesion - mean_healthy), 2)
        else:
            # Too few pixels to trust measurement
            self.results['hsv_shift'] = 0

    # 7. Edge Roughness Index

    def analyze_edge_roughness(self):
        """
        Returns:
        - perimeter/area ratio
        - solidity (area / convex hull area)
        - mean Sobel gradient (edge texture)
        """
        mask = self.detect_leaf_mask()

        # --- Contour-based metrics (perimeter/area, solidity) ---
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"error": "No leaf contour found"}

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        roughness_ratio = perimeter / area if area != 0 else 0
        solidity = area / hull_area if hull_area != 0 else 0

        # --- Sobel gradient (edge sharpness) ---
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(magnitude)

        # return {
        #     "perimeter_area_ratio": round(roughness_ratio, 4),
        #     "solidity": round(solidity, 4),
        #     "mean_sobel_gradient": round(mean_gradient, 2)
        # }

        self.results['edge_roughness'] = "rough" if round(
            solidity, 2) < 0.90 else "smooth"

    # 8. Curl Index: width to height ratio
    def analyze_curl_index(self):
        mask = self.detect_leaf_mask()
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.results['curl_index'] = 0
            return
        cnt = max(contours, key=cv2.contourArea)

        # Fit rotated rectangle
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        aspect_ratio = min(w, h) / max(w, h)

        # Perimeter to convex hull perimeter (curl index)
        perimeter = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_perimeter = cv2.arcLength(hull, True)
        curl_index = perimeter / hull_perimeter if hull_perimeter != 0 else 0

        self.results['curl_index'] = curl_index

    # 9. Texture Entropy using GLCM
    def analyze_texture_entropy(self):
        """
        Calculate the texture entropy using GLCM.
        """
        glcm = graycomatrix(self.gray, distances=[5], angles=[
                            0], levels=256, symmetric=True, normed=True)
        self.results['texture_entropy'] = graycoprops(glcm, 'contrast')[0, 0]

    # 10. Internode Length
    def analyze_internode_length(self):
        """
        Analyze internode length.
        """
        # Placeholder for internode length analysis
        # This would typically require manual measurement or a specific algorithm
        self.results['internode_length'] = "Manual measurement required"

    # 11. Green Intensity and Branch Count
    def analyze_green_intensity_and_branch_count(self):
        """
        Analyze green intensity and branch count.
        """
        # Extract the leaf mask
        mask = self.detect_leaf_mask()

        # Split the image into BGR channels
        b, g, r = cv2.split(self.image)

        # Use the mask to extract green values from leaf only
        green_pixels = g[mask > 0]

        # Compute mean green value
        mean_green = np.mean(green_pixels) if green_pixels.size > 0 else 0

        # Create visual masked image
        masked_leaf = cv2.bitwise_and(self.image, self.image, mask=mask)
        masked_rgb = cv2.cvtColor(masked_leaf, cv2.COLOR_BGR2RGB)
        self.results['mean_green_intensity'] = mean_green
        self.results['branch_count'] = "Manual or skeleton-based count"

    # 12. Speck Count
    def analyze_speck_count(self):
        """
        Count the number of specks in the image.
        """
        speck_mask = cv2.inRange(self.gray, 0, 30)
        speck_contours, _ = cv2.findContours(
            speck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.results['speck_count'] = len(
            [c for c in speck_contours if cv2.contourArea(c) > 5])

    # 13. Rust Area (%)
    def analyze_rust_area(self):
        """
        Calculate the percentage of rust area.
        """
        # Define HSV range for rust-colored regions
        lower_rust = np.array([5, 100, 30])
        upper_rust = np.array([20, 255, 150])

        # Create rust mask
        rust_mask = cv2.inRange(self.hsv, lower_rust, upper_rust)

        # Create rust overlay
        rust_overlay = cv2.bitwise_and(self.image, self.image, mask=rust_mask)

        # Calculate rust area percent relative to image area
        rust_area_percent = (np.sum(rust_mask > 0) /
                             (self.image.shape[0] * self.image.shape[1])) * 100

        rust_rgb = cv2.cvtColor(rust_overlay, cv2.COLOR_BGR2RGB)

        self.results['rust_area_percent'] = round(rust_area_percent, 2)

    # 14. Pustule Count
    def analyze_pustule_count(self):
        """
        Count the number of pustules.
        """
        # Leaf mask to eliminate background
        leaf_mask = self.detect_leaf_mask()

        # Rust color mask to identify pustules
        rust_mask = cv2.inRange(self.hsv, (5, 100, 100), (20, 255, 255))

        # Apply leaf mask to rust mask
        rust_on_leaf = cv2.bitwise_and(rust_mask, rust_mask, mask=leaf_mask)

        # Find contours of rust-colored regions on leaf
        rust_contours, _ = cv2.findContours(
            rust_on_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count only valid-sized pustules
        valid_pustules = [c for c in rust_contours if cv2.contourArea(c) > 10]

        # Draw bounding boxes
        result = self.image.copy()
        for c in valid_pustules:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Draw contours
        result_rgb_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        self.results['pustule_count'] = len(valid_pustules)

    # 15. Leaf Droop Angle
    def analyze_droop_angle(self):
        """
        Estimate the leaf droop angle.
        """
        self.results['droop_angle'] = "Requires keypoint estimation or manual measurement"

    # Process the leaf region
    def detect_leaf_mask(self):
        """Returns a binary mask of the leaf region (green/brown tones)."""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Detect broad leaf color range
        leaf_mask = cv2.inRange(hsv, (15, 40, 40), (90, 255, 255))
        return leaf_mask
