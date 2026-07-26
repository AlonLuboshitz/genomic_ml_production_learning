"""
Cloud storage module for the genomics ML pipeline.

This module provides two classes:
- CloudStorage: Low-level S3 wrapper (upload, download, list)
- StorageManager: High-level interface that reads config and handles the
  local vs S3 toggle automatically

Requirements:
    - boto3 must be installed (pip install boto3)
    - AWS credentials configured via ~/.aws/credentials or env vars
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib

from genomics_ml.utils.logging import get_logger

logger = get_logger("genomics_ml.storage")


class CloudStorage:
    """
    Low-level wrapper around boto3 S3 client.

    Handles the actual S3 operations: upload, download, list.
    This class doesn't know about configs or toggles — it just
    talks to S3 when told to.
    """

    def __init__(self, bucket: str | None = None):
        """
        Initialize the S3 client.

        Args:
            bucket: S3 bucket name. Falls back to S3_BUCKET env var,
                    then defaults to "genomics-ml-artifacts".
        """
        self.bucket = bucket or os.getenv("S3_BUCKET", "genomics-ml-artifacts")
        # boto3.client creates a connection to AWS S3
        # Credentials come from ~/.aws/credentials or environment variables
        self.s3 = None  # lazy-loaded to avoid errors if boto3 isn't installed
        self._initialized = False

    def _get_client(self):
        """Lazily create the S3 client (only when actually needed)."""
        if not self._initialized:
            try:
                import boto3

                self.s3 = boto3.client("s3")
                self.s3.head_bucket(Bucket=self.bucket)
                print(f"Connected to S3 bucket: {self.bucket}")
                self._initialized = True
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 storage. "
                    "Install it with: pip install boto3"
                )
        return self.s3

    def upload(self, local_path: str, s3_key: str | None = None) -> bool:
        """
        Upload a local file to S3.

        Args:
            local_path: Path to the file on your machine (e.g., "models/baseline.pkl")
            s3_key: Where to store it in S3. Defaults to just the filename.

        Returns:
            True if successful, False otherwise.
        """
        s3_key = s3_key or Path(local_path).name
        try:
            client = self._get_client()
            client.upload_file(local_path, self.bucket, s3_key)
            print(f"Uploaded {local_path} -> s3://{self.bucket}/{s3_key}")
            return True
        except OSError as e:
            print(f"Upload failed: {e}")
            return False

    def download(self, s3_key: str, local_path: str | None = None) -> bool:
        """
        Download a file from S3 to your local machine.

        Args:
            s3_key: The file's key (path) in S3.
            local_path: Where to save it locally. Defaults to the filename.

        Returns:
            True if successful, False otherwise.
        """
        local_path = local_path or Path(s3_key).name
        try:
            client = self._get_client()
            client.download_file(self.bucket, s3_key, local_path)
            print(f"Downloaded s3://{self.bucket}/{s3_key} -> {local_path}")
            return True
        except OSError as e:
            print(f"Download failed: {e}")
            return False

    def list_objects(self, prefix: str = "") -> list:
        """
        List files in an S3 bucket (optionally filtered by prefix).

        Args:
            prefix: Filter to files starting with this path (e.g., "models/").

        Returns:
            List of S3 keys (file paths).
        """
        try:
            client = self._get_client()
            resp = client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except OSError as e:
            print(f"List failed: {e}")
            return []


class StorageManager:
    """
    High-level storage interface that handles local + optional cloud saves.

    Reads config at init to determine backend. Provides save() and load()
    methods that always work locally, and optionally sync to S3.

    Usage:
        config = load_config()
        storage = StorageManager(config)
        storage.save(model, "models/baseline.pkl")
        model = storage.load("models/baseline.pkl")
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the storage manager from pipeline config.

        Args:
            config: Full pipeline config dict. If None, defaults to local-only.
        """
        self.cloud = None
        if config is None:
            return
        backend = config.get("storage", {}).get("backend", "local")
        bucket = config.get("storage", {}).get("bucket", "genomics-ml-artifacts")
        if backend.lower() == "s3":
            self.cloud = CloudStorage(bucket)

    def save(self, model, local_path: str) -> None:
        """
        Save a model locally, and optionally sync to cloud.

        Always saves locally first. If S3 is enabled, also uploads.

        Args:
            model: The model object to serialize (sklearn pipeline, etc.)
            local_path: Where to save locally (e.g., "models/baseline.pkl").
        """
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        joblib.dump(model, local_path)
        if self.cloud:
            self.cloud.upload(local_path)

    def load(self, local_path: str):
        """
        Load a model from local disk, fetching from cloud if needed.

        If the local file doesn't exist but cloud is enabled,
        downloads it from S3 first.

        Args:
            local_path: Path to the local model file.

        Returns:
            The deserialized model object.
        """
        if not Path(local_path).exists():
            if self.cloud is None:
                raise RuntimeError(
                    f"No model in local path: {local_path} AND NO CLOUD USAGE"
                )
            else:
                logger.info(
                    f"model: {local_path} not found locally, downloading from S3..."
                )
                self.cloud.download(Path(local_path).name, local_path)
            if not Path(local_path).exists():
                raise RuntimeError("Couldnt download file from s3 cloud")
        model = joblib.load(local_path)
        logger.info(f"Loaded model: {local_path}")
        return model
