import os
import requests

MODEL_URLS = {
    "urlset_ml_assets/feature_columns.pkl": "https://drive.google.com/uc?export=download&id=1wD4hoRkVTFbbTj8If0jrysggam81__-P",
    "urlset_ml_assets/processed_data.pkl": "https://drive.google.com/uc?export=download&id=1IsEw3O6zabRk3qYC55rR6SlpeGhTonNT",
    "urlset_ml_assets/scaler.pkl": "https://drive.google.com/uc?export=download&id=1gqUozBDu_HhqjdUMI1WTWrKagXrfk76d",
    "urlset_ml_assets/urlset_ensemble_model.pkl": "https://drive.google.com/uc?export=download&id=1jAg6R2OJYq3WgqyDHYmnYeA_JsHwozTO",
}

def download_file(dest, url):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{dest} downloaded.")
    else:
        print(f"{dest} already exists, skipping.")

if __name__ == "__main__":
    for dest, url in MODEL_URLS.items():
        if url.startswith("<YOUR_LINK_FOR_"):
            print(f"Please update download_models.py with actual download links for {dest}")
        else:
            download_file(dest, url)
