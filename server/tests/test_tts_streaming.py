#!/usr/bin/env python3
"""
TTS 스트리밍 기능 테스트 스크립트
AWS Polly Neural 엔진이 제대로 작동하는지 확인합니다.
"""

import boto3
import asyncio
from core.config import settings

# Polly 클라이언트 초기화
polly_client = boto3.client('polly', region_name=settings.AWS_REGION)


async def test_polly_streaming():
    """Polly Neural 엔진 테스트"""
    test_text = "안녕하세요. AWS Polly Neural 엔진을 사용한 실시간 스트리밍 테스트입니다."

    print(f"테스트 텍스트: {test_text}")
    print("AWS Polly 호출 중...")

    try:
        # Polly synthesize_speech 호출 (비동기로 실행)
        response = await asyncio.to_thread(
            polly_client.synthesize_speech,
            VoiceId='Seoyeon',
            Engine='neural',  # Neural 엔진 사용
            OutputFormat='mp3',
            Text=test_text
        )

        print("✓ Polly 호출 성공!")

        # AudioStream 확인
        audio_stream = response.get('AudioStream')

        if audio_stream:
            print("✓ AudioStream 획득 성공!")

            # 청크 단위로 읽기 테스트
            chunk_size = 1024
            total_bytes = 0
            chunk_count = 0

            while True:
                chunk = await asyncio.to_thread(audio_stream.read, chunk_size)

                if not chunk:
                    break

                total_bytes += len(chunk)
                chunk_count += 1

            await asyncio.to_thread(audio_stream.close)

            print(f"✓ 스트리밍 완료!")
            print(f"  - 총 청크 수: {chunk_count}")
            print(f"  - 총 바이트: {total_bytes}")
            print(f"  - 청크 크기: {chunk_size} bytes")

            # 오디오 파일 저장 (선택사항)
            print("\n오디오 파일을 저장하시겠습니까? (y/n): ", end="")

        else:
            print("✗ AudioStream을 찾을 수 없습니다.")

    except Exception as e:
        print(f"✗ 에러 발생: {e}")
        print(f"에러 타입: {type(e).__name__}")

        # 일반적인 에러 해결 방법 안내
        if "InvalidSsmlException" in str(e):
            print("\n해결 방법: 텍스트에 특수 문자나 SSML 태그가 잘못 포함되었을 수 있습니다.")
        elif "AccessDeniedException" in str(e):
            print("\n해결 방법: IAM 역할에 Polly 접근 권한이 있는지 확인하세요.")
        elif "InvalidParameterException" in str(e):
            print("\n해결 방법: VoiceId 'Seoyeon'이 Neural 엔진을 지원하는지 확인하세요.")


async def test_polly_voices():
    """사용 가능한 한국어 음성 목록 확인"""
    print("\n=== 사용 가능한 한국어 음성 ===")

    try:
        response = await asyncio.to_thread(
            polly_client.describe_voices,
            LanguageCode='ko-KR'
        )

        for voice in response.get('Voices', []):
            print(f"- {voice['Name']} ({voice['Gender']})")
            print(f"  지원 엔진: {', '.join(voice.get('SupportedEngines', []))}")

    except Exception as e:
        print(f"✗ 음성 목록 조회 실패: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("AWS Polly TTS 스트리밍 테스트")
    print("=" * 60)
    print()

    # 비동기 실행
    asyncio.run(test_polly_streaming())

    # 사용 가능한 음성 확인
    asyncio.run(test_polly_voices())

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
