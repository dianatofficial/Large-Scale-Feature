import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class MinIOStorageClient:
    """Client for interacting with S3/MinIO object store in distributed pipelines."""

    def __init__(self, endpoint: str = "localhost:9000", access_key: str = "minioadmin", secret_key: str = "minioadmin", secure: bool = False):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from minio import Minio
                self._client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure
                )
            except ImportError:
                logger.warning("minio package not installed.")
                return None
        return self._client

    def ensure_bucket_exists(self, bucket_name: str) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info("Created MinIO bucket: %s", bucket_name)
            return True
        except Exception as exc:
            logger.error("Failed to verify/create MinIO bucket: %s", exc)
            return False

    def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            self.ensure_bucket_exists(bucket_name)
            client.fput_object(bucket_name, object_name, file_path)
            logger.info("Uploaded %s to minio://%s/%s", file_path, bucket_name, object_name)
            return True
        except Exception as exc:
            logger.error("Failed to upload file to MinIO: %s", exc)
            return False
