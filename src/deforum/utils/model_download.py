import os
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from deforum.utils.logging_config import logger


def get_filename_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    return filename


def download_file(
    download_url: str, destination: str, filename: str, force_download: bool = False, token: str = None
):
    # Ensure download url is a string
    assert isinstance(download_url, str), "Download URL should be a string."

    # Ensure destination exists
    os.makedirs(destination, exist_ok=True)

    # Construct the full file path
    filepath = os.path.join(destination, filename)

    # Check if the file already exists
    if os.path.exists(filepath):
        if not force_download:
            logger.info(
                f"File {filename} already exists at {destination}. Use force_download=True to re-download."
            )
            return filename
        else:
            logger.info(
                f"File {filename} already exists at {destination}. Re-downloading as per request."
            )

    # Download the file with a progress bar
    logger.info(f"Downloading {filename}...")
    response = requests.get(f'{download_url}{token}', stream=True, headers={'Content-Disposition': 'attachment'})
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filepath, "wb") as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    # Check for possible download errors
    if total_size != 0 and t.n != total_size:
        logger.info("ERROR: Something went wrong while downloading the file.")
    else:
        logger.info(f"{filename} downloaded successfully!")

    return filename


def download_from_civitai(
    model_id: str, destination: str, force_download: bool = False
):
    # Ensure model id is a string
    assert isinstance(model_id, str), "Model ID should be a string."

    # Fetch the model details
    response = requests.get(f"https://civitai.com/api/v1/models/{model_id}")
    response.raise_for_status()

    model_data = response.json()
    download_url = model_data["modelVersions"][0]["downloadUrl"]
    filename = model_data["modelVersions"][0]["files"][0]["name"]

    # civitai token required for download
    civitai_token = '?token=a44763d416db87cfb4fdb6b70369f4a3'

    # Use the helper function to download the model
    return download_file(download_url, destination, filename, force_download, token=civitai_token)


def download_from_civitai_by_version_id(
    model_id: str, destination: str, force_download: bool = False
):
    # Ensure model id is a string
    assert isinstance(model_id, str), "Model ID should be a string."
    civitai_token = '?token=a44763d416db87cfb4fdb6b70369f4a3'
    # Fetch the model details
    response = requests.get(f"https://civitai.com/api/v1/model-versions/{model_id}")
    response.raise_for_status()

    model_data = response.json()

    for key, value in model_data.items():
        print(key)

        if key.strip().lower() == 'files':
            for k in value:
                print(f"             {k}")
                if k['type'].lower() == 'model':
                    download_url = k["downloadUrl"]
                    filename = k["name"]

                    return download_file(download_url, destination, filename, force_download, token=civitai_token)


    return None