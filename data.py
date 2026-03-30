import os
from pathlib import Path
from typing import List
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

DATASET_PAGE_CANDIDATES = [
    "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
    "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html",
]

OUTPUT_DIR = "data/sample_videos"
DEFAULT_LIMIT = 70
REQUEST_TIMEOUT = 10
CHUNK_SIZE = 1024

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_video_links(session: requests.Session, limit: int = DEFAULT_LIMIT) -> List[str]:
    print("Fetching dataset page...")
    response = None

    for candidate_url in DATASET_PAGE_CANDIDATES:
        try:
            response = session.get(candidate_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            print(f"Using dataset page: {candidate_url}")
            break
        except requests.RequestException as exc:
            print(f"Failed to reach {candidate_url}: {exc}")

    if response is None:
        raise requests.ConnectionError(
            "Unable to reach any dataset page URL. Check internet/DNS or update URLs."
        )

    soup = BeautifulSoup(response.text, "html.parser")

    fall_links = []
    adl_links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "cam0-rgb.zip" in href:
            full_url = urljoin(response.url, href)

            if "fall-" in href:
                fall_links.append(full_url)
            elif "adl-" in href:
                adl_links.append(full_url)

    # Remove duplicates and sort for deterministic runs.
    fall_links = sorted(list(set(fall_links)))
    adl_links = sorted(list(set(adl_links)))

    # Balanced selection with odd limits and category shortages handled gracefully.
    fall_target = min(len(fall_links), (limit + 1) // 2)
    adl_target = min(len(adl_links), limit // 2)
    selected = fall_links[:fall_target] + adl_links[:adl_target]

    remaining = limit - len(selected)
    if remaining > 0:
        extra_falls = fall_links[fall_target:]
        extra_adls = adl_links[adl_target:]
        selected.extend((extra_falls + extra_adls)[:remaining])

    return selected


def download_files(session: requests.Session, urls: List[str]) -> None:
    for url in urls:
        filename = url.split("/")[-1]
        if "adl-" in filename:
            base_dir = "data/adl"
            path = os.path.join(base_dir, filename)
        elif "fall-" in filename:
            # Extract the number, e.g., fall-01-cam0-rgb.zip -> 01
            parts = filename.split("-")
            num = parts[1]
            seq_dir = f"fall-{int(num):02d}"
            base_dir = os.path.join("data/falls", seq_dir)
            os.makedirs(base_dir, exist_ok=True)
            path = os.path.join(base_dir, filename)
        else:
            continue  # Skip if not adl or fall

        if os.path.exists(path):
            print(f"Skipping existing file: {filename}")
            continue

        print(f"Downloading: {filename}")

        try:
            r = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()

            with open(path, "wb") as f:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            print(f"Saved: {filename}")
            
            # Extract the zip file
            extract_to = os.path.dirname(path)
            with ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            os.remove(path)
            print(f"Extracted and removed zip: {filename}")
        except requests.RequestException as e:
            print(f"Error: {e}")


def main() -> None:
    session = build_session()
    urls = fetch_video_links(session=session, limit=DEFAULT_LIMIT)
    print(f"\nFound {len(urls)} valid dataset files\n")

    download_files(session=session, urls=urls)

    print(f"\nDone downloading up to {DEFAULT_LIMIT} files.")


if __name__ == "__main__":
    main()
