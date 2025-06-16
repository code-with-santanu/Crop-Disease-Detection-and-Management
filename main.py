import image_analysis
import pandas as pd

import os

report = image_analysis.analysis_report(
    'F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\images')


# Create a DataFrame from the results
df = pd.DataFrame(report)

# Export the DataFrame to a CSV file
output_dir = "F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\output"
output_path = os.path.join(
    output_dir, "image_analysis_results.csv")
df.to_csv(output_path, index=False)


print("Analysis report generated successfully.")

# print(result)
