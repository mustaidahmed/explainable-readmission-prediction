import os
import zipfile
import urllib.request

def download_and_extract():
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    zip_path = "dataset.zip"
    extract_dir = "data"

    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

    print(f"Extracting zip to '{extract_dir}'...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")
        
        # Check files inside data
        extracted_files = os.listdir(extract_dir)
        print(f"Extracted files: {extracted_files}")
        
        # Check if there is another zip file in there (like dataset_diabetes.zip)
        for f in extracted_files:
            if f.endswith(".zip") or f == "dataset_diabetes.zip":
                file_path = os.path.join(extract_dir, f)
                print(f"Extracting nested zip {f}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                os.remove(file_path)
                print(f"Extracted and removed {f}")
                
        print(f"Final files in '{extract_dir}': {os.listdir(extract_dir)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

if __name__ == "__main__":
    download_and_extract()
