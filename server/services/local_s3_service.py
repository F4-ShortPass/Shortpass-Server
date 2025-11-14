# server/services/local_s3_service.py
"""
로컬 파일 시스템을 사용하여 S3 동작을 시뮬레이션하는 서비스.
실제 AWS S3 대신 로컬에 로그 파일을 저장하여 개발을 용이하게 합니다.
"""
import os
import json
from pathlib import Path

class LocalS3Service:
    """로컬 파일 시스템에 로그를 저장하는 S3 시뮬레이션 서비스"""

    def __init__(self, base_path: str = "server/local_s3_storage"):
        """
        서비스를 초기화합니다.

        Args:
            base_path: 로그가 저장될 기본 루트 디렉토리
        """
        self.base_path = Path(base_path)
        # 기본 경로가 존재하지 않으면 생성
        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"🗂️  LocalS3Service initialized. Storage path: {self.base_path.resolve()}")

    def save_json_log(self, data: dict, s3_key: str) -> str:
        """
        주어진 데이터를 JSON 파일로 변환하여 지정된 S3 키 경로에 저장합니다.

        Args:
            data (dict): 저장할 딕셔너리 데이터.
            s3_key (str): S3 객체 키와 동일한 형식의 파일 경로.
                         예: "company/1/job/2/applicant/3/interview/4/v1-pipeline/01_transcript.json"

        Returns:
            str: 저장된 파일의 전체 로컬 경로
        
        Raises:
            IOError: 파일 쓰기 실패 시
        """
        try:
            # s3_key를 로컬 파일 시스템 경로로 변환
            local_path = self.base_path / s3_key
            
            # 파일이 위치할 디렉토리가 존재하지 않으면 생성
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # JSON 데이터를 파일에 쓰기
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Log saved locally: {local_path.resolve()}")
            return str(local_path.resolve())

        except (IOError, TypeError) as e:
            print(f"✗ Failed to save log to {s3_key}: {e}")
            raise IOError(f"Failed to write log file for key {s3_key}") from e

    def save_binary_log(self, data: bytes, s3_key: str) -> str:
        """
        주어진 바이너리 데이터를 파일로 지정된 S3 키 경로에 저장합니다.

        Args:
            data (bytes): 저장할 바이너리 데이터 (예: 오디오 파일 내용).
            s3_key (str): S3 객체 키와 동일한 형식의 파일 경로.

        Returns:
            str: 저장된 파일의 전체 로컬 경로
        
        Raises:
            IOError: 파일 쓰기 실패 시
        """
        try:
            local_path = self.base_path / s3_key
            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, 'wb') as f:
                f.write(data)
            
            print(f"✅ Binary log saved locally: {local_path.resolve()}")
            return str(local_path.resolve())

        except IOError as e:
            print(f"✗ Failed to save binary log to {s3_key}: {e}")
            raise IOError(f"Failed to write binary log file for key {s3_key}") from e

    def get_log_path(self, s3_key: str) -> str:
        """
        주어진 S3 키에 해당하는 전체 로컬 파일 경로를 반환합니다.
        """
        return str((self.base_path / s3_key).resolve())

# 싱글톤 인스턴스
local_s3_service = LocalS3Service()
