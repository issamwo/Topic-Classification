from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    storage_bucket_name: str
    source_blob_name: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    unpreprocessed_data_path: Path
    column_text: str
    column_topic: str
    cleaned_data_path: Path



@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    cleaned_data_path: Path
    preprocessed_spilitted_data_path: Path
    max_words: float
    topic_names: dict
    test_size: float