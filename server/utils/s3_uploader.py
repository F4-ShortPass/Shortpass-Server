"""
Simple local S3 uploader stub for dev/test.

Copies the given file into server/local_s3_storage (optionally namespaced
by folder) and returns a file://-style path that downstream code can use
as a URL placeholder.
"""

import shutil
from pathlib import Path
from uuid import uuid4


def upload_file_and_get_url(file_path: str, folder: str = "") -> str:
    base_dir = Path(__file__).resolve().parent.parent / "local_s3_storage"
    target_dir = base_dir / folder if folder else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    source = Path(file_path)
    unique_name = f"{uuid4().hex}_{source.name}"
    dest = target_dir / unique_name

    shutil.copyfile(source, dest)
    return dest.as_posix()
