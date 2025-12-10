"""
S3 서비스 선택기: USE_AWS_S3 환경변수에 따라 AWS S3 또는 로컬 S3 시뮬레이터를 반환합니다.
"""
from core import config
from services.local_s3_service import LocalS3Service


def get_s3_service():
    """
    Returns:
        AwsS3Service | LocalS3Service: 선택된 S3 서비스 인스턴스
    """
    if config.USE_AWS_S3:
        # 지연 로딩하여 로컬 개발 시 boto3 미설치 문제를 피함
        from services.aws_s3_service import AwsS3Service
        return AwsS3Service()
    return LocalS3Service()
