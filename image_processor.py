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
        Calculate the average shape irregularity of greasy lesions.
        """
        _, thresh = cv2.threshold(self.gray, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_irregularity = [cv2.arcLength(
            c, True) ** 2 / (4 * np.pi * cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 30]
        self.results['avg_irregularity'] = np.mean(
            shape_irregularity) if shape_irregularity else 0

    # 6. HSV Color Shift
    def analyze_hsv_shift(self):
        """
        Calculate the HSV color shift in lesions.
        """
        lesion_mask = cv2.inRange(self.gray, 0, 80)
        mean_hsv = cv2.mean(self.hsv, mask=lesion_mask)[:3]
        self.results['hsv_shift'] = np.linalg.norm(
            np.array(mean_hsv) - np.array([60, 60, 60]))

    # 7. Edge Roughness Index
    def analyze_edge_roughness(self):
        """
        Calculate the edge roughness index.
        """
        edges = cv2.Canny(self.gray, 100, 200)
        total_area = self.image.shape[0] * self.image.shape[1]
        roughness = cv2.countNonZero(edges) / total_area
        self.results['edge_roughness'] = roughness

    # 8. Curl Index: width to height ratio
    def analyze_curl_index(self):
        """
        Calculate the curl index (width to height ratio).
        """
        _, thresh = cv2.threshold(self.gray, 60, 255, cv2.THRESH_BINARY_INV)
        x, y, w, h = cv2.boundingRect(thresh)
        self.results['curl_index'] = round(w / h, 0) if h != 0 else 0

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
        mean_green = np.mean(self.image[:, :, 1])
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
        rust_mask = cv2.inRange(self.hsv, (5, 100, 100), (20, 255, 255))
        rust_area = np.sum(rust_mask > 0)
        total_area = self.image.shape[0] * self.image.shape[1]
        self.results['rust_area_percent'] = (rust_area / total_area) * 100

    # 14. Pustule Count
    def analyze_pustule_count(self):
        """
        Count the number of pustules.
        """
        rust_mask = cv2.inRange(self.hsv, (5, 100, 100), (20, 255, 255))
        rust_contours, _ = cv2.findContours(
            rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.results['pustule_count'] = len(
            [c for c in rust_contours if cv2.contourArea(c) > 10])

    # 15. Leaf Droop Angle
    def analyze_droop_angle(self):
        """
        Estimate the leaf droop angle.
        """
        self.results['droop_angle'] = "Requires keypoint estimation or manual measurement"
