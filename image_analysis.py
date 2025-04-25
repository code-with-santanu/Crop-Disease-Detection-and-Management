from image_processor import ImageProcessor
import cv2
import os
import pandas as pd


def analyze_image(image_path):
    # Initialize the ImageProcessor
    processor = ImageProcessor(image_path)

    # get the file name without extension
    file_name = os.path.basename(image_path).split('.')[0]
    image_id = file_name

    # Load the image
    processor.load_image()

    # Process the image using ImageProcessor methods
    processor.analyze_lesion_count()
    processor.analyze_avg_spot_diameter()
    processor.analyze_ring_count()
    processor.analyze_yellowing_area()
    processor.analyze_shape_irregularity()
    processor.analyze_hsv_shift()
    processor.analyze_edge_roughness()
    processor.analyze_curl_index()
    processor.analyze_texture_entropy()
    processor.analyze_internode_length()
    processor.analyze_green_intensity_and_branch_count()
    processor.analyze_speck_count()
    processor.analyze_rust_area()
    processor.analyze_pustule_count()
    processor.analyze_droop_angle()

    # Get the results
    result = processor.get_result()
    result['image_id'] = image_id
    # Assuming the disease is part of the filename
    result['disease'] = image_id.split('_')[0]

    return result


def analysis_report(directory_path):

    # Prepare a list to store results
    report = []

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory_path, filename)
            image_id = os.path.splitext(filename)[0]

            # Read the image
            image = cv2.imread(image_path)

            # Process the image and get the results
            result = analyze_image(image_path)

            # Append the result to the results list
            report.append(result)

    # Create a DataFrame from the results
    df = pd.DataFrame(report)

    # Export the DataFrame to a CSV file
    output_dir = "F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\output"
    output_path = os.path.join(
        output_dir, "image_analysis_results.csv")
    df.to_csv(output_path, index=False)
