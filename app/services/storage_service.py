import requests
import os
from azure.storage.blob import BlobServiceClient, BlobClient
from urllib.parse import urlparse
from typing import Tuple


def download_blob(blob_url: str) -> bytes:
    response = requests.get(blob_url, timeout=30)
    response.raise_for_status()
    return response.content


def _extract_blob_name(blob_url: str) -> str:
    """
    Extract only the blob name from a blob URL.
    Example:
        https://acc.blob.core.windows.net/mri-scans/study123.png
    Returns:
        study123.png
    """
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")  # mri-scans/study123.png
    _, blob_name = path.split("/", 1)
    return blob_name


def upload_segmented_image(
    image_bytes: bytes,
    original_blob_url: str,
    suffix: str = "_segmented.png",
    segmentation_container: str = "segmentation"
) -> str:

    # Extract blob name from original URL
    original_blob_name = _extract_blob_name(original_blob_url)
    print(f"Original blob name: {original_blob_name}")
    
    # Build segmented file name
    if "." in original_blob_name:
        base, _ = original_blob_name.rsplit(".", 1)
        new_blob_name = base + suffix
    else:
        new_blob_name = original_blob_name + suffix

    new_blob_name = new_blob_name.split('mriscan/')[-1]

    # Get BlobServiceClient
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")

    blob_service = BlobServiceClient.from_connection_string(conn_str)

    # Upload to the 'segmentation' container
    blob_client = blob_service.get_blob_client(
        container=f"scan/{segmentation_container}",
        blob=new_blob_name
    )

    blob_client.upload_blob(image_bytes, overwrite=True)

    # Return the publicly accessible URL
    return blob_client.url
