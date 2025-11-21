#!/usr/bin/env python3
"""
JD PDF 전처리 스크립트
AWS Bedrock 대신 OpenAI API를 사용하여 정적 파일 생성

사용법:
    python preprocess_jd.py

환경변수:
    OPENAI_API_KEY: OpenAI API 키 (필수)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# PDF 파싱 라이브러리
try:
    import pypdf
except ImportError:
    print("❌ pypdf가 설치되지 않았습니다. 설치 명령어: pip install pypdf")
    sys.exit(1)

# OpenAI 라이브러리
try:
    from openai import OpenAI
except ImportError:
    print("❌ openai가 설치되지 않았습니다. 설치 명령어: pip install openai")
    sys.exit(1)


class JDPreprocessor:
    """JD PDF 전처리 클래스"""

    def __init__(self, pdf_path: str, output_path: str):
        self.pdf_path = Path(pdf_path)
        self.output_path = Path(output_path)

        # OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
                "설정 방법: export OPENAI_API_KEY='your-api-key'"
            )

        self.client = OpenAI(api_key=api_key)

        # 고정 공통 역량 (기존 코드와 동일)
        self.COMMON_COMPETENCIES = [
            "고객지향",
            "도전정신",
            "협동",
            "팀워크",
            "목표지향",
            "책임감"
        ]

    def extract_text_from_pdf(self) -> str:
        """PDF에서 텍스트 추출"""
        try:
            print(f"\n📄 PDF 파일 읽기: {self.pdf_path}")

            if not self.pdf_path.exists():
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {self.pdf_path}")

            reader = pypdf.PdfReader(str(self.pdf_path))
            text_parts = []

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text_parts.append(page_text)
                print(f"  ✓ 페이지 {i+1}/{len(reader.pages)} 추출 완료")

            full_text = "\n\n".join(text_parts)
            print(f"\n✅ 총 {len(full_text)} 글자 추출 완료\n")

            return full_text

        except Exception as e:
            print(f"❌ PDF 파싱 실패: {e}")
            raise

    def extract_competencies_with_openai(self, jd_text: str) -> Dict[str, Any]:
        """OpenAI API를 사용하여 역량 추출"""
        try:
            print("🤖 OpenAI API 호출 중... (GPT-4o)")

            prompt = f"""
다음은 채용공고(Job Description) 문서입니다. 이 문서를 분석하여 다음 정보를 JSON 형식으로 추출해주세요.

<채용공고>
{jd_text[:8000]}
</채용공고>

요구사항:
1. **core_competencies**: 이 직무에서 가장 중요한 핵심 역량 5개 (2-4글자 명사형)
2. **job_competencies**: 직무 관련 역량 6개 (기술적 역량, 소프트 스킬 포함)
3. **company_name**: 회사명 추출 (없으면 "Unknown Company")
4. **job_title**: 직무명 추출 (예: "Product Manager", "Backend Developer" 등)
5. **system_prompt**: 이 JD를 기반으로 면접을 진행할 AI 면접관의 페르소나를 정의하는 시스템 프롬프트 (200-300자)
6. **persona_summary**: 2명의 면접관 페르소나 정의 (각각 다른 평가 초점)

응답 형식 (JSON):
{{
  "company_name": "삼성물산",
  "job_title": "Product Manager",
  "core_competencies": [
    "글로벌 시장 분석",
    "프로젝트 관리",
    "협상력",
    "공급망 이해",
    "리스크 관리"
  ],
  "job_competencies": [
    "데이터분석",
    "문제해결력",
    "커뮤니케이션",
    "창의적 사고",
    "기술적 이해",
    "리더십"
  ],
  "system_prompt": "당신은 글로벌 트레이딩 분야의 15년 경력 시니어 면접관입니다. 지원자의 시장 분석 능력, 프로젝트 관리 경험, 그리고 글로벌 비즈니스 감각을 중점적으로 평가합니다. 구체적인 경험과 성과 수치를 바탕으로 STAR 기법을 활용한 질문을 통해 지원자의 역량을 객관적으로 판단합니다.",
  "persona_summary": [
    {{
      "type": "전략적 사고형 면접관",
      "focus": "시장 분석 및 전략 수립 능력 평가",
      "target_competencies": ["글로벌 시장 분석", "데이터분석", "문제해결력"],
      "example_question": "복잡한 시장 상황에서 어떻게 데이터를 분석하고 전략을 수립했나요?"
    }},
    {{
      "type": "실행력 중심형 면접관",
      "focus": "프로젝트 실행 및 협업 능력 평가",
      "target_competencies": ["프로젝트 관리", "협상력", "커뮤니케이션"],
      "example_question": "다양한 이해관계자와 협업하여 프로젝트를 완수한 경험을 말씀해주세요."
    }}
  ]
}}

반드시 유효한 JSON 형식으로만 응답해주세요. 추가 설명은 넣지 마세요.
"""

            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 채용공고 분석 전문가입니다. 항상 정확한 JSON 형식으로 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            # JSON 파싱
            result = self._parse_json_response(response_text)

            # 공통 역량 추가
            result["common_competencies"] = self.COMMON_COMPETENCIES

            print("✅ OpenAI API 응답 파싱 완료\n")

            return result

        except Exception as e:
            print(f"❌ OpenAI API 호출 실패: {e}")
            raise

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """OpenAI 응답에서 JSON 추출 및 파싱"""
        try:
            # JSON 코드 블록 제거 (```json ... ``` 형식)
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            # JSON 파싱
            result = json.loads(response_text)

            # 필수 필드 검증
            required_fields = [
                "core_competencies",
                "job_competencies",
                "system_prompt",
                "company_name",
                "job_title",
                "persona_summary"
            ]

            for field in required_fields:
                if field not in result:
                    raise ValueError(f"필수 필드 누락: {field}")

            # core_competencies 개수 검증
            if len(result["core_competencies"]) != 5:
                print(f"⚠️  경고: core_competencies가 5개가 아닙니다 ({len(result['core_competencies'])}개)")
                result["core_competencies"] = result["core_competencies"][:5]

            # job_competencies 개수 검증
            if len(result["job_competencies"]) != 6:
                print(f"⚠️  경고: job_competencies가 6개가 아닙니다 ({len(result['job_competencies'])}개)")
                result["job_competencies"] = result["job_competencies"][:6]

            return result

        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            print(f"응답 내용:\n{response_text[:500]}...")
            raise

    def save_to_json(self, data: Dict[str, Any]) -> None:
        """추출된 데이터를 JSON 파일로 저장"""
        try:
            # 출력 디렉토리 생성
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # JSON 저장
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✅ JSON 파일 저장 완료: {self.output_path}")
            print(f"   파일 크기: {self.output_path.stat().st_size} bytes\n")

        except Exception as e:
            print(f"❌ JSON 저장 실패: {e}")
            raise

    def run(self) -> None:
        """전체 전처리 실행"""
        try:
            print("="*60)
            print("JD PDF 전처리 시작")
            print("="*60)

            # 1. PDF 텍스트 추출
            jd_text = self.extract_text_from_pdf()

            # 2. OpenAI API로 역량 추출
            persona_data = self.extract_competencies_with_openai(jd_text)

            # 3. JSON 저장
            self.save_to_json(persona_data)

            # 4. 결과 출력
            print("="*60)
            print("전처리 완료!")
            print("="*60)
            print(f"\n 추출된 정보:")
            print(f"  - 회사명: {persona_data['company_name']}")
            print(f"  - 직무: {persona_data['job_title']}")
            print(f"  - 핵심 역량: {', '.join(persona_data['core_competencies'])}")
            print(f"  - 직무 역량: {', '.join(persona_data['job_competencies'])}")
            print(f"  - 공통 역량: {', '.join(persona_data['common_competencies'])}")
            print(f"\n✅ 서버 시작 시 이 파일이 자동으로 로드됩니다.")

        except Exception as e:
            print(f"\n❌ 전처리 실패: {e}")
            sys.exit(1)


def main():
    """메인 함수"""
    # 경로 설정
    pdf_path = "docs/jd.pdf"
    output_path = "assets/persona_data.json"

    # 전처리 실행
    preprocessor = JDPreprocessor(pdf_path, output_path)
    preprocessor.run()


if __name__ == "__main__":
    main()
