# Integrating datasets with Storage blobs

To upload a dataset to azure blob, `from encortex.azure_utils import upload_to_azure, download_from_azure`. Use `upload_to_azure(...)` to upload datasets to a container where files are stored as blobs.
In practice, we store everything under the `datasets` container.