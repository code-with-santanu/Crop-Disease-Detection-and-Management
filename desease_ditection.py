import image_analysis
import pandas as pd
import detect_tomato_disease as dt

import os

# reports = image_analysis.analysis_report(
#     'F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\images\\testing')
reports = image_analysis.analysis_report(
    'F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\images\\5771_sample')


# # Create a DataFrame from the results
# df = pd.DataFrame(report)
output = []

for report in reports:
    res = {}
    # print(report)
    res['image_id'] = report['image_id']
    disease = dt.detect_tomato_disease(report)
    res['disease'] = disease

    output.append(res)


ans = pd.DataFrame(output)
# print(ans)


# Export the DataFrame to a CSV file
output_dir = "F:\\Users\\Santanu\\Desktop\\paper_syn\\codebase\\output"
output_path = os.path.join(
    output_dir, "5771_sample.csv")
ans.to_csv(output_path, index=False)
print("done")
