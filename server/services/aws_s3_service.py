"""
AWS S3 구현체. LocalS3Service와 동일한 인터페이스(save_json_log, save_binary_log, get_log_path)를 제공합니다.
환경변수 USE_AWS_S3=true 일 때 사용합니다.
"""
import json
from typing import Optional
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from core import config


class AwsS3Service:
    def __init__(self, bucket_name: Optional[str] = None, region_name: Optional[str] = None):
        self.bucket = bucket_name or config.S3_BUCKET_NAME
        self.region = region_name or config.AWS_REGION
        # boto3는 자격 증명을 환경/메타데이터에서 자동으로 로드
        self.client = boto3.client("s3", region_name=self.region)

    def save_json_log(self, data: dict, s3_key: str) -> str:
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.client.put_object(Bucket=self.bucket, Key=s3_key, Body=body, ContentType="application/json; charset=utf-8")
            return f"s3://{self.bucket}/{s3_key}"
        except (BotoCoreError, ClientError) as e:
            raise IOError(f"Failed to upload JSON log to s3://{self.bucket}/{s3_key}: {e}") from e

    def save_binary_log(self, data: bytes, s3_key: str) -> str:
        try:
            self.client.put_object(Bucket=self.bucket, Key=s3_key, Body=data)
            return f"s3://{self.bucket}/{s3_key}"
        except (BotoCoreError, ClientError) as e:
            raise IOError(f"Failed to upload binary log to s3://{self.bucket}/{s3_key}: {e}") from e

    def get_log_path(self, s3_key: str) -> str:
        return f"s3://{self.bucket}/{s3_key}"
