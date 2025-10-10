#!/usr/bin/env python3
"""
OpenAI TTS 및 Whisper 테스트 스크립트
"""

import asyncio
from openai import OpenAI
from core.config import settings

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=settings.OPENAI_API_KEY)


async def test_openai_tts():
    """OpenAI TTS 테스트"""
    print("\n" + "="*60)
    print("OpenAI TTS 테스트")
    print("="*60)

    test_text = "안녕하세요. OpenAI TTS를 사용한 음성 합성 테스트입니다."
    print(f"테스트 텍스트: {test_text}")

    try:
        print("OpenAI TTS API 호출 중...")

        def generate_speech():
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=test_text,
                response_format="mp3"
            )
            return response

        response = await asyncio.to_thread(generate_speech)

        # 오디오 바이트 확인
        audio_bytes = response.content
        print(f"✓ TTS 생성 성공!")
        print(f"  - 오디오 크기: {len(audio_bytes)} bytes")
        print(f"  - 형식: MP3")

        # 오디오 파일 저장 (선택사항)
        save = input("\n오디오 파일을 저장하시겠습니까? (y/n): ")
        if save.lower() == 'y':
            output_file = "test_tts_output.mp3"
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            print(f"✓ 파일 저장됨: {output_file}")

        return True

    except Exception as e:
        print(f"✗ TTS 에러: {e}")
        return False


async def test_openai_whisper():
    """OpenAI Whisper STT 테스트"""
    print("\n" + "="*60)
    print("OpenAI Whisper STT 테스트")
    print("="*60)

    # 먼저 TTS로 오디오 생성
    print("테스트용 오디오 생성 중...")
    test_text = "이것은 음성 인식 테스트입니다."

    try:
        # TTS로 오디오 생성
        def generate_speech():
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=test_text,
                response_format="mp3"
            )
            return response

        response = await asyncio.to_thread(generate_speech)
        audio_bytes = response.content

        # 임시 파일로 저장
        temp_file = "temp_audio.mp3"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        print(f"✓ 테스트 오디오 생성 완료 ({len(audio_bytes)} bytes)")
        print(f"원본 텍스트: {test_text}")

        # Whisper로 변환
        print("\nWhisper API 호출 중...")

        def transcribe_audio():
            with open(temp_file, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="ko",
                    response_format="json"
                )
            return result.text

        transcribed_text = await asyncio.to_thread(transcribe_audio)

        print(f"✓ 음성 인식 성공!")
        print(f"  - 인식된 텍스트: {transcribed_text}")

        # 임시 파일 삭제
        import os
        os.remove(temp_file)

        return True

    except Exception as e:
        print(f"✗ Whisper 에러: {e}")
        return False


async def test_available_voices():
    """사용 가능한 TTS 음성 목록"""
    print("\n" + "="*60)
    print("OpenAI TTS 사용 가능한 음성")
    print("="*60)

    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    print("사용 가능한 음성:")
    for voice in voices:
        print(f"  - {voice}")

    print("\n현재 설정: nova (여성 목소리)")
    print("변경하려면 interview_service.py의 voice 파라미터를 수정하세요.")


async def main():
    print("\n" + "="*60)
    print("OpenAI TTS & Whisper 통합 테스트")
    print("="*60)

    # API 키 확인
    if not OPENAI_API_KEY:
        print("✗ OPENAI_API_KEY가 설정되지 않았습니다!")
        print("  .env 파일에 OPENAI_API_KEY를 추가하세요.")
        return

    print(f"✓ API 키 확인됨: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-5:]}")

    # TTS 테스트
    tts_result = await test_openai_tts()

    # Whisper 테스트
    whisper_result = await test_openai_whisper()

    # 사용 가능한 음성
    await test_available_voices()

    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    print(f"TTS 테스트: {'✓ 성공' if tts_result else '✗ 실패'}")
    print(f"Whisper 테스트: {'✓ 성공' if whisper_result else '✗ 실패'}")

    if tts_result and whisper_result:
        print("\n✓ 모든 테스트 통과! 면접 서비스를 시작할 수 있습니다.")
        print("\n서버 실행:")
        print("  cd /home/ec2-user/flex/server")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("\n✗ 일부 테스트 실패. 에러 메시지를 확인하세요.")

    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
