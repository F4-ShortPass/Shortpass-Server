import boto3
import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. Bedrock 클라이언트 생성 ---
try:
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime', 
        region_name='us-east-1' # 👈 본인의 Bedrock 리전
    )
    print("Boto3 클라이언트 생성 성공.")
    
except Exception as e:
    print(f"Boto3 클라이언트 생성 실패: {e}")
    sys.exit(1)


# --- 2. 호출할 모델과 프롬프트 정의 (Claude 3 Sonnet으로 변경) ---
model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0") # 👈 모델 ID 설정
prompt = "AWS Bedrock이 무엇인지 한국어로 3줄 요약해줘."

print(f"모델({model_id}) 호출 시작...")

# Claude 3 Sonnet이 요구하는 요청 본문(body) 형식
# (Llama와 형식이 다릅니다!)
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31", # 👈 Claude 전용 버전 명시
    "max_tokens": 512,
    "temperature": 0.5,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
})


# --- 3. Bedrock 모델 호출 ---
try:
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    # --- 4. 응답 파싱 및 출력 (Claude 3 형식에 맞게 변경) ---
    response_body_str = response['body'].read().decode('utf-8')
    response_body_json = json.loads(response_body_str)

    # Claude 3의 응답은 'content' 리스트의 첫 번째 'text'에 있습니다.
    generation_text = response_body_json['content'][0]['text']

    print("--- Bedrock 응답 ---")
    print(generation_text)
    print("--------------------")
    print(f"Stop Reason: {response_body_json['stop_reason']}")


except boto3.exceptions.botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'AccessDeniedException':
        print(f"!!! 에러: AccessDeniedException.")
        print("IAM Role에 Bedrock 접근 권한이 없거나, AWS 콘솔에서 'Claude 3 Sonnet' 모델 접근 활성화를 하지 않았습니다.")
    else:
        print(f"!!! Boto3 에러: {error}")

except Exception as e:
    print(f"!!! 알 수 없는 에러: {e}")