import re
import pandas as pd
import os


def analyse_result():

    # Load the uploaded CSV file
    file_path = "F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\output\\5771_sample.csv"
    df = pd.read_csv(file_path)

    # Display the first few rows and column names to understand the structure
    df.head(), df.columns.tolist()

    # Function to remove trailing digits from image_id

    def remove_trailing_numbers(s):
        return re.sub(r'_\d+$', '', s)

    # Apply the function to create a new column 'base_id'
    df['base_id'] = df['image_id'].apply(remove_trailing_numbers)

    # Convert stringified list in 'disease' to actual list
    df['disease_list'] = df['disease'].apply(eval)

    # Check if base_id exists in disease_list
    df['match'] = df.apply(lambda row: 1 if row['base_id']
                           in row['disease_list'] else 0, axis=1)

    # Display the updated DataFrame
    df.head()

    # Export the DataFrame to a CSV file
    output_dir = "F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\output"
    output_path = os.path.join(
        output_dir, "5771_sample_analysis.csv")
    df.to_csv(output_path, index=False)
    print("Analysis report generated successfully with base_id and match status.")


def calculate_accuracy():
    # Load the analysis results
    file_path = "F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\output\\5771_sample_analysis.csv"
    df = pd.read_csv(file_path)

    correct_matches = df[df['match'] > 0]
    print(correct_matches.head())
    print(correct_matches.shape)
    # Export the DataFrame to a CSV file
    output_dir = "F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\output"
    output_path = os.path.join(
        output_dir, "correct_matches.csv")
    correct_matches.to_csv(output_path, index=False)

    early_blight_matches = correct_matches[correct_matches['base_id']
                                           == 'early_blight']
    bacterial_spot_matches = correct_matches[correct_matches['base_id']
                                             == 'bacterial_spot']
    leaf_curl_matches = correct_matches[correct_matches['base_id']
                                        == 'leaf_curl']
    rust_matches = correct_matches[correct_matches['base_id'] == 'rust']
    healthy_matches = correct_matches[correct_matches['base_id'] == 'healthy']

    overall_accuracy = (len(correct_matches)/5771)*100
    early_blight_accuracy = (
        len(early_blight_matches) / 1000) * 100
    bacterial_spot_accuracy = (
        len(bacterial_spot_matches) / 1000) * 100
    leaf_curl_accuracy = (len(leaf_curl_matches) /
                          1000) * 100
    rust_accuracy = (len(rust_matches) /
                     1771) * 100
    healthy_accuracy = (len(healthy_matches) /
                        1000) * 100

    print(f"Accuracy of desease_detection: {overall_accuracy:.2f}%")
    print(f"Accuracy of early_blight: {early_blight_accuracy:.2f}%")
    print(f"Accuracy of bacterial_spot: {bacterial_spot_accuracy:.2f}%")
    print(f"Accuracy of leaf_curl: {leaf_curl_accuracy:.2f}%")
    print(f"Accuracy of rust: {rust_accuracy:.2f}%")
    print(f"Accuracy of healthy: {healthy_accuracy:.2f}%")


if __name__ == "__main__":
    # analyse_result()
    calculate_accuracy()
