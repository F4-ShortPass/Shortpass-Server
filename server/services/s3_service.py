# server/services/s3_service.py
"""
S3 파일 업로드/다운로드 서비스
"""
import boto3
from botocore.exceptions import ClientError
from typing import BinaryIO, Optional
import io
from datetime import datetime
from core.config import settings


class S3Service:
    """S3 파일 관리 서비스"""

    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
        self.bucket_name = settings.S3_BUCKET_NAME

    def upload_file(
        self,
        file_content: bytes,
        file_name: str,
        folder: str = "jd_pdfs"
    ) -> str:
        """
        파일을 S3에 업로드

        Args:
            file_content: 파일 바이너리 내용
            file_name: 파일명
            folder: S3 폴더 경로 (기본: jd_pdfs)

        Returns:
            str: S3 객체 키 (예: "jd_pdfs/2024/10/29/company_jd.pdf")

        Raises:
            Exception: 업로드 실패 시
        """
        try:
            # 날짜별 폴더 구조 생성
            date_path = datetime.now().strftime("%Y/%m/%d")
            s3_key = f"{folder}/{date_path}/{file_name}"

            # S3 업로드
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType='application/pdf'
            )

            print(f"✓ File uploaded to S3: {s3_key}")
            return s3_key

        except ClientError as e:
            print(f"✗ S3 upload failed: {e}")
            raise Exception(f"Failed to upload file to S3: {str(e)}")

    def download_file(self, s3_key: str) -> bytes:
        """
        S3에서 파일 다운로드

        Args:
            s3_key: S3 객체 키

        Returns:
            bytes: 파일 내용

        Raises:
            Exception: 다운로드 실패 시
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            file_content = response['Body'].read()
            print(f"✓ File downloaded from S3: {s3_key}")
            return file_content

        except ClientError as e:
            print(f"✗ S3 download failed: {e}")
            raise Exception(f"Failed to download file from S3: {str(e)}")

    def get_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        S3 파일의 presigned URL 생성 (다운로드용)

        Args:
            s3_key: S3 객체 키
            expiration: URL 유효 시간 (초, 기본 1시간)

        Returns:
            str: Presigned URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"✗ Failed to generate presigned URL: {e}")
            raise Exception(f"Failed to generate presigned URL: {str(e)}")

    def delete_file(self, s3_key: str) -> bool:
        """
        S3에서 파일 삭제

        Args:
            s3_key: S3 객체 키

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            print(f"✓ File deleted from S3: {s3_key}")
            return True
        except ClientError as e:
            print(f"✗ S3 delete failed: {e}")
            return False

    def file_exists(self, s3_key: str) -> bool:
        """
        S3에 파일이 존재하는지 확인

        Args:
            s3_key: S3 객체 키

        Returns:
            bool: 파일 존재 여부
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except ClientError:
            return False
