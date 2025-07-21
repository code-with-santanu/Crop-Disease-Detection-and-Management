import os


def rename_files_in_directory(directory, prefix="file", start_index=1):
    try:
        files = os.listdir(directory)
        for index, filename in enumerate(files, start=start_index):
            old_path = os.path.join(directory, filename)
            if os.path.isfile(old_path):
                file_extension = os.path.splitext(filename)[1]
                new_name = f"{prefix}_{index}{file_extension}"
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
        print("Files renamed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Replace with your directory path
directory_path = "F://Users//Santanu//Desktop//paper_syn//tomato-leaves//healthy"
rename_files_in_directory(directory_path, prefix="healthy")
