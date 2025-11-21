# server/ai/parsers/persona_question_parser.py
"""
페르소나 질문 PDF 파싱
기업 관계자들이 답해야 할 필수 질문들을 PDF에서 추출합니다.
"""
import pdfplumber
import io
import json
import boto3
from typing import List, Dict, Any, Optional
import re
from core.config import settings


class PersonaQuestionParser:
    """
    페르소나 질문 PDF 파서

    PDF에서 면접 질문들을 추출하고, LLM을 사용하여 구조화된 질문 데이터로 변환합니다.
    """

    def __init__(self):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=settings.AWS_REGION)
        self.model_id = settings.BEDROCK_MODEL_ID

    def parse_pdf(self, pdf_content: bytes) -> str:
        """
        PDF에서 텍스트 추출

        Args:
            pdf_content: PDF 파일 바이너리 내용

        Returns:
            str: 추출된 전체 텍스트

        Raises:
            Exception: PDF 파싱 실패 시
        """
        try:
            full_text = []

            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                print(f"📄 PDF 로드: {len(pdf.pages)} 페이지")

                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text:
                        cleaned_text = self._clean_text(text)
                        full_text.append(cleaned_text)
                        print(f"  페이지 {page_num}: {len(cleaned_text)} 문자")

            result = "\n\n".join(full_text)
            print(f"✓ 총 추출된 텍스트: {len(result)} 문자")

            return result

        except Exception as e:
            print(f"❌ PDF 파싱 실패: {e}")
            raise Exception(f"Failed to parse PDF: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def extract_questions_with_llm(self, pdf_text: str, company_name: str) -> Dict[str, Any]:
        """
        LLM을 사용하여 PDF 텍스트에서 구조화된 질문 데이터 추출

        Args:
            pdf_text: PDF에서 추출한 텍스트
            company_name: 회사명

        Returns:
            Dict containing:
                - questions: List[Dict] - 추출된 질문 리스트
                - persona_info: Dict - 페르소나 메타데이터
        """
        prompt = f"""
다음은 {company_name}의 면접관 페르소나를 만들기 위한 PDF 문서입니다.
이 문서에서 면접 질문들과 페르소나 정보를 추출해주세요.

[PDF 텍스트]:
{pdf_text}

다음 JSON 형식으로 응답해주세요:

{{
  "persona_info": {{
    "persona_name": "면접관 이름 또는 역할 (예: '기술 면접관', 'HR 면접관')",
    "archetype": "analytical|supportive|stress_tester 중 하나 (문서의 톤에 따라)",
    "description": "페르소나에 대한 간단한 설명 (1-2문장)",
    "focus_areas": ["집중 영역1", "집중 영역2"]
  }},
  "questions": [
    {{
      "question_text": "질문 내용",
      "question_type": "technical|behavioral|situational|cultural 중 하나",
      "expected_keywords": ["기대되는", "키워드", "리스트"],
      "evaluation_criteria": ["평가 기준1", "평가 기준2"],
      "difficulty_level": 1-5 사이의 정수
    }}
  ]
}}

중요사항:
1. 질문은 명확하게 구분되어야 합니다
2. 질문이 명시적으로 없으면 문서 내용을 기반으로 적절한 질문을 생성하세요
3. question_type은 질문의 성격에 따라 분류하세요
4. expected_keywords는 좋은 답변에 포함될 것으로 예상되는 키워드입니다
5. evaluation_criteria는 답변을 평가할 기준입니다
6. difficulty_level은 질문의 난이도입니다 (1=쉬움, 5=어려움)
7. JSON만 응답하고 다른 텍스트는 포함하지 마세요
"""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3  # 구조화된 출력을 위해 낮은 temperature
            })

            print("🤖 LLM으로 질문 추출 중...")

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read().decode('utf-8'))
            llm_output = response_body['content'][0]['text'].strip()

            # JSON 추출 (LLM이 추가 텍스트를 포함할 수 있으므로)
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                llm_output = json_match.group(0)

            result = json.loads(llm_output)

            print(f"✓ {len(result.get('questions', []))}개의 질문 추출 완료")

            return result

        except json.JSONDecodeError as e:
            print(f"❌ LLM 응답 JSON 파싱 실패: {e}")
            print(f"LLM 출력: {llm_output}")
            # 기본값 반환
            return self._extract_questions_fallback(pdf_text, company_name)

        except Exception as e:
            print(f"❌ LLM 질문 추출 실패: {e}")
            # Fallback: 간단한 규칙 기반 추출
            return self._extract_questions_fallback(pdf_text, company_name)

    def _extract_questions_fallback(self, pdf_text: str, company_name: str) -> Dict[str, Any]:
        """
        LLM 실패 시 폴백: 간단한 규칙 기반 질문 추출
        """
        print("⚠️  Fallback: 규칙 기반 질문 추출")

        # 질문 패턴 찾기 (숫자. 또는 Q: 로 시작하는 라인)
        question_patterns = [
            r'^\d+\.\s+(.+?)(?=\n\d+\.|\Z)',  # 1. 질문형태
            r'^Q\d*[:)]\s+(.+?)(?=\nQ\d*[:)]|\Z)',  # Q: 또는 Q1: 형태
            r'^\?\s+(.+?)(?=\n\?|\Z)',  # ? 로 시작
        ]

        questions = []
        for pattern in question_patterns:
            matches = re.finditer(pattern, pdf_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                question_text = match.group(1).strip()
                if len(question_text) > 10:  # 너무 짧은 텍스트 제외
                    questions.append({
                        "question_text": question_text,
                        "question_type": "general",
                        "expected_keywords": [],
                        "evaluation_criteria": ["답변의 명확성", "논리적 구조"],
                        "difficulty_level": 3
                    })

        # 질문을 찾지 못한 경우, 전체 텍스트를 하나의 질문 세트로 처리
        if not questions:
            questions.append({
                "question_text": f"{company_name}에서 요구하는 역량에 대해 설명해주세요",
                "question_type": "general",
                "expected_keywords": [],
                "evaluation_criteria": ["답변의 명확성", "경험의 구체성"],
                "difficulty_level": 3
            })

        return {
            "persona_info": {
                "persona_name": f"{company_name} 면접관",
                "archetype": "analytical",
                "description": f"{company_name}의 채용 기준을 평가하는 면접관",
                "focus_areas": ["역량 평가"]
            },
            "questions": questions
        }

    def parse_persona_questions(
        self,
        pdf_content: bytes,
        company_name: str
    ) -> Dict[str, Any]:
        """
        페르소나 질문 PDF를 파싱하여 구조화된 데이터 반환

        Args:
            pdf_content: PDF 파일 바이너리
            company_name: 회사명

        Returns:
            Dict containing:
                - full_text: 원본 텍스트
                - persona_info: 페르소나 메타데이터
                - questions: 추출된 질문 리스트
        """
        print(f"\n{'='*60}")
        print(f"페르소나 질문 PDF 파싱 시작: {company_name}")
        print(f"{'='*60}\n")

        # 1. PDF에서 텍스트 추출
        full_text = self.parse_pdf(pdf_content)

        # 2. LLM으로 질문 추출
        extracted_data = self.extract_questions_with_llm(full_text, company_name)

        result = {
            "full_text": full_text,
            "persona_info": extracted_data.get("persona_info", {}),
            "questions": extracted_data.get("questions", [])
        }

        print(f"\n{'='*60}")
        print(f"✓ 파싱 완료")
        print(f"  - 질문 수: {len(result['questions'])}")
        print(f"  - 페르소나: {result['persona_info'].get('persona_name', 'N/A')}")
        print(f"{'='*60}\n")

        return result
