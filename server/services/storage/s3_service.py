"""
S3 저장소 서비스
- Full Transcript 저장/조회 (Claim Check Pattern)
- Agent 실행 로그 저장
"""
import boto3
import json
from typing import Dict, Any, List, Optional

class S3Service:
    def __init__(self, bucket_name: str, region_name: str = 'ap-northeast-2'):
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.bucket_name = bucket_name

    def upload_json(self, key: str, data: Dict[str, Any]) -> str:
        """
        Uploads a JSON object to S3.
        
        Args:
            key (str): The S3 object key (path).
            data (Dict[str, Any]): The JSON data to upload.
            
        Returns:
            str: The S3 URI of the uploaded object.
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                ContentType='application/json; charset=utf-8'
            )
            return f"s3://{self.bucket_name}/{key}"
        except Exception as e:
            print(f"Error uploading JSON to S3: {e}")
            raise

    def download_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Downloads a JSON object from S3.
        
        Args:
            key (str): The S3 object key (path).
            
        Returns:
            Optional[Dict[str, Any]]: The downloaded JSON data, or None if not found.
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except self.s3_client.exceptions.NoSuchKey:
            print(f"Object with key '{key}' not found in bucket '{self.bucket_name}'.")
            return None
        except Exception as e:
            print(f"Error downloading JSON from S3: {e}")
            raise
            
    def save_agent_log(self, log_data: Dict[str, Any], log_id: str) -> str:
        """
        Saves agent execution logs to S3.
        
        Args:
            log_data (Dict[str, Any]): The log data to save.
            log_id (str): A unique identifier for the log (e.g., agent_run_id, timestamp).
            
        Returns:
            str: The S3 URI of the saved log.
        """
        key = f"agent_logs/{log_id}.json"
        return self.upload_json(key, log_data)
        
    def get_agent_log(self, log_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves agent execution logs from S3.
        
        Args:
            log_id (str): The unique identifier for the log.
            
        Returns:
            Optional[Dict[str, Any]]: The retrieved log data, or None if not found.
        """
        key = f"agent_logs/{log_id}.json"
        return self.download_json(key)
