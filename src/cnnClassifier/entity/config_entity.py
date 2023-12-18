from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    storage_bucket_name: str
    source_blob_name: str
    local_data_file: Path
    unzip_dir: Path