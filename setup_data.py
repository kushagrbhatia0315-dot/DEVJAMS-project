import os
import shutil
import kagglehub
print("Download in progress")
wf_path = kagglehub.dataset_download("rtatman/188-million-us-wildfires")
print(f"Wildfires downloaded to: {wf_path}")
for root, dirs, files in os.walk(wf_path):
    for file in files:
        if file.endswith(".sqlite"):
            src = os.path.join(root, file)
            dst = "FPA_FOD_20170508.sqlite"
            shutil.copy(src, dst)
            print(f"-> Copied wildfire database to root folder.")
storms_path = kagglehub.dataset_download("shacharcohen/usa-storms-19962019")
print(f"Storms downloaded to: {storms_path}")
target_dir = os.path.join("data", "Storms 1996-2019")
os.makedirs(target_dir, exist_ok=True)
count = 0
for root, dirs, files in os.walk(storms_path):
    for file in files:
        if file.endswith(".csv"):
            src = os.path.join(root, file)
            dst = os.path.join(target_dir, file)
            shutil.copy(src, dst)
            count += 1
print(f"-> Copied {count} storm CSV files into {target_dir}")
print("Data setup is now complete. You can now run python app.py")
