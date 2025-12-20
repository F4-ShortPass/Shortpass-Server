import boto3
import time
import requests # Transcribe 결과 다운로드용
import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 이제 'core' 패키지를 절대 경로로 임포트할 수 있습니다.
from core.config import settings

# --- 1. (필수) 설정 ---
S3_BUCKET_NAME = settings.S3_BUCKET_NAME
AWS_REGION = settings.AWS_REGION
LOCAL_AUDIO_FILE = "test_polly_output.mp3"

# --- 2. AWS 클라이언트 생성 ---
try:
    polly_client = boto3.client('polly', region_name=AWS_REGION)
    transcribe_client = boto3.client('transcribe', region_name=AWS_REGION)
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    print("AWS 클라이언트 생성 성공.")
except Exception as e:
    print(f"클라이언트 생성 실패: {e}")
    sys.exit(1)

# --- 3. (Polly) 텍스트를 음성 파일(MP3)로 변환 ---
try:
    test_text = "안녕하세요. 자기소개를 해 주세요."
    print(f"\n[Polly] 텍스트를 MP3로 변환 시작: '{test_text}'")
    
    response = polly_client.synthesize_speech(
        Text=test_text,
        OutputFormat='mp3',
        VoiceId='Seoyeon', # 한국어 여성 음성
        Engine='neural'
    )

    # 음성 스트림을 로컬 파일에 저장
    audio_stream = response.get('AudioStream')
    if audio_stream:
        with open(LOCAL_AUDIO_FILE, 'wb') as file:
            file.write(audio_stream.read())
        audio_stream.close()
        print(f"[Polly] 음성 파일 저장 완료: {LOCAL_AUDIO_FILE}")
    else:
        raise Exception("Polly 오디오 스트림을 받지 못했습니다.")

except Exception as e:
    print(f"!!! Polly 작업 실패: {e}")
    sys.exit(1)

# --- 4. (S3) 생성된 MP3 파일을 S3에 업로드 ---
try:
    s3_key = f"audio-tests/{LOCAL_AUDIO_FILE}"
    s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"
    print(f"\n[S3] 파일을 S3로 업로드 시작: {s3_uri}")
    
    s3_client.upload_file(LOCAL_AUDIO_FILE, S3_BUCKET_NAME, s3_key)
    
    print("[S3] 업로드 완료.")

except Exception as e:
    print(f"!!! S3 업로드 실패: {e}")
    sys.exit(1)

# --- 5. (Transcribe) S3 파일을 텍스트로 변환 ---
try:
    # Transcribe 작업 이름은 계정 내에서 고유해야 함
    job_name = f"MyTestJob-{int(time.time())}"
    print(f"\n[Transcribe] 변환 작업 시작 (Job: {job_name})")
    
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode='ko-KR', # 한국어
        Media={'MediaFileUri': s3_uri}
    )

    # --- 6. Transcribe 작업 완료까지 대기 (Polling) ---
    final_text = "Transcribe 작업 실패"
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        
        if job_status in ['COMPLETED', 'FAILED']:
            print(f"[Transcribe] 작업 상태: {job_status}")
            
            if job_status == 'COMPLETED':
                # 결과 파일(JSON) URL 가져오기
                result_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                
                # JSON 결과 다운로드 및 파싱
                result_response = requests.get(result_uri)
                result_json = result_response.json()
                final_text = result_json['results']['transcripts'][0]['transcript']
            
            # (정리) Transcribe 작업 삭제
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            break # Polling 루프 종료
            
        print(f"[Transcribe] 작업 진행 중 ({job_status})... 5초 대기")
        time.sleep(5)

    # --- 7. 최종 결과 출력 ---
    print("\n--- 최종 결과 ---")
    print(f"Polly 원본 텍스트: {test_text}")
    print(f"Transcribe 변환 텍스트: {final_text}")

except Exception as e:
    print(f"!!! Transcribe 작업 실패: {e}")

finally:
    # (정리) S3 파일 삭제
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        print(f"[S3] S3 파일({s3_key}) 삭제 완료.")
    except Exception as e:
        print(f"!!! S3 파일 삭제 실패: {e}")
