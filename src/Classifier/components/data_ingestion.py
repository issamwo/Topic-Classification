import os
import urllib.request as request
import zipfile
from Classifier import logger
from Classifier.utils.common import get_size
from Classifier.entity.config_entity import DataIngestionConfig


from google.oauth2 import service_account

# Path to the service account file
service_account_file = "C:/Users/frup00090410/Downloads/sonic-name-403220-e74704f4f952.json"

# Create a Credentials object from the service account file
credentials = service_account.Credentials.from_service_account_file(
    service_account_file
)

from google.cloud import storage

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self) -> str:
        """Downloads a blob from the bucket."""
        try:
            # The ID of your GCS bucket
            # bucket_name = "your-bucket-name"
            # The ID of your GCS object
            # source_blob_name = "storage-object-name"
            # The path to which the file should be downloaded

            bucket_name = self.config.storage_bucket_name
            source_blob_name = self.config.source_blob_name
            destination_file_name = self.config.local_data_file

            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            storage_client = storage.Client(credentials=credentials)
            bucket = storage_client.bucket(bucket_name)
            # Construct a client side representation of a blob.
            # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
            # any content from Google Cloud Storage. As we don't need additional data,
            # using `Bucket.blob` is preferred here.
            blob = bucket.blob(source_blob_name)
            logger.info(f"downloading data from {bucket} into file {destination_file_name}")

            blob.download_to_filename(destination_file_name)
            logger.info(f"downloaded data from {bucket} into file {destination_file_name}")
        except Exception as e:
            raise e


    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)