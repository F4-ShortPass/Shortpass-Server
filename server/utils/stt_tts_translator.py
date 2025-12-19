# 251121 채아
# 면접 세션에서 사용되는 stt, tts 로직 분리한 유틸리티 파일입니당 ~.~
# utils/stt_tts_translator.py

import boto3
import os
import uuid
import time
import requests
from contextlib import closing

from core.config import settings
from utils.s3_uploader import upload_file_and_get_url

class SttTtsTranslator:
    def __init__(self):
        self.region = settings.AWS_REGION
        self.bucket_name = settings.S3_BUCKET_NAME
        
        # AWS 클라이언트 초기화 (Polly, Transcribe)
        try:
            self.polly = boto3.client('polly', region_name=self.region)
            self.transcribe = boto3.client('transcribe',region_name=self.region)
            print("✅ [Translator] AWS Polly & Transcribe 연결 성공")
        except Exception as e:
            print(f"❌ [Translator] 초기화 실패: {e}")
        
    # 준비된 질문 텍스트 s3에 업로드 -> s3 url 반환
    def text_to_audio(self, text: str, folder: str) -> str:
        """
        텍스트를 받아 Polly로 MP3를 생성하고, S3에 업로드한 뒤 Presigned URL을 반환
        Args:
            text (str): 변환할 텍스트
            folder (str): S3 내 저장할 폴더 경로 (예: interview_41)
        """
        # 고유한 파일명 생성 (충돌 방지)
        file_id = uuid.uuid4()
        filename = f"{file_id}.mp3"
        local_path = f"./{filename}" # 현재 위치에 임시 생성

        try:
            # Polly 요청 및 로컬 저장
            response = self.polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId='Seoyeon',
                Engine='neural'
            )

            if "AudioStream" in response:
                with closing(response["AudioStream"]) as stream:
                    with open(local_path, "wb") as file:
                        file.write(stream.read())
            else:
                print("❌ [TTS] polly 응답 오류로 MP3 생성 실패")
                return None

            # S3 업로드
            url = upload_file_and_get_url(local_path, folder=folder)

            return url

        except Exception as e:
            print(f"❌ [TTS] 처리 중 에러 발생: {e}")
            return None
        finally:
            # 로컬 임시 파일 삭제 (Clean up)
            if os.path.exists(local_path):
                os.remove(local_path)

    # 받은 답변 audio s3에 업로드 -> text로 변환하여 리턴
    def audio_to_text(self, local_path: str, folder: str) -> str:
        """
        Args:
            local_path (str): 변환할 로컬 오디오 파일 경로
            folder (str): S3 내 저장할 폴더 경로 (예: interview_41)
        """

        if not os.path.exists(local_path):
            print(f"❌ [STT] 로컬 파일 찾을 수 없음: {local_path}")
            return None
        
        try:
            # S3에 오디오 파일 업로드
            s3_url = upload_file_and_get_url(local_path, folder=folder)

            # 받은 s3_url을 transcribe 요청하기 위해 파싱
            s3_key = None
            if "amazonaws.com/" in s3_url:
                key_part = s3_url.split("amazonaws.com/")[1]
                s3_key = key_part.split("?")[0]
            
            if not s3_key:
                print("❌ [STT] S3 URL 파싱 실패. Transcribe 요청 불가.")
                return None
            
            audio_url = f"s3://{self.bucket_name}/{s3_key}"
            job_name = f"stt_job_{uuid.uuid4()}"

            # Transcribe 작업 시작
            self.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': audio_url},
                MediaFormat='mp3', 
                LanguageCode='ko-KR'
            )

            # 작업 완료될때까지 대기
            while True:
                status = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(0.5)

            # 변환된 텍스트 받아오기
            if job_status == 'COMPLETED':
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                response = requests.get(transcript_uri)
                data = response.json()
                transcript = data['results']['transcripts'][0]['transcript']
                
                return transcript
            else:
                print(f"❌ [STT] 변환 실패: {status['TranscriptionJob'].get('FailureReason')}")
                return None

        except Exception as e:
            print(f"❌ [STT] 처리 중 에러: {e}")
            return None

stt_tts_translator = SttTtsTranslator()
