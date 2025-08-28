import os
import multiprocessing
import requests

import pandas as pd

def download_photo(url_save_tuple):
    url, save_path = url_save_tuple
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"404 Not Found: {url}")
            return None
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {url}")
        return save_path
    except Exception as e:
        print(f"Failed: {url} ({e})")
        return None

def main(urls):
    os.makedirs('./image', exist_ok=True)
    # Generate save paths for each URL
    url_save_tuples = []
    for idx, url in enumerate(urls):
        ext = os.path.splitext(url)[1]
        if not ext or len(ext) > 5:
            ext = '.jpg'
        save_path = os.path.join('./image', f"img_{idx}{ext}")
        url_save_tuples.append((url, save_path))

    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
        results = pool.map(download_photo, url_save_tuples)
    print(f"Downloaded {sum(1 for r in results if r)} images.")

# TODO: Read URLs from CSV and call main
if __name__ == "__main__":
    df = pd.read_csv("data/image-index-NDK.final.csv")
    urls = df["urls"].dropna().tolist()[:200]
    main(urls)
