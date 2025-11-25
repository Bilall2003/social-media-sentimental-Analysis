# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

# print("Path to dataset files:", path)

import kagglehub
import os

path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
print("Downloaded path:", path)

print("\nFiles inside the dataset folder:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
