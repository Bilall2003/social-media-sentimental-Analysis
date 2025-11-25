import kagglehub

# Download latest version
# Choose Test.csv or Train.csv as per your requirement
path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

print("Path to dataset files:", path)