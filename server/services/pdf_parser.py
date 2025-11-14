"""
PDF 파서
PDF에서 텍스트를 추출하고 기업 정보를 분석합니다.
"""

import os
import json
import boto3
from PyPDF2 import PdfReader
from typing import List
from models.company_profile import CompanyProfile
from core.config import settings


class PDFParser:
    """PDF 파싱 및 기업 정보 추출"""

    def __init__(self):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=settings.AWS_REGION)
        self.model_id = settings.BEDROCK_MODEL_ID

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF에서 텍스트 추출

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            추출된 텍스트
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() + "\n"

            print(f"PDF 텍스트 추출 완료: {len(text)} 자")
            return text

        except Exception as e:
            print(f"PDF 파싱 에러: {e}")
            return ""

    def analyze_jd_with_bedrock(self, jd_text: str, pdf_filename: str) -> dict:
        """
        Bedrock을 사용하여 JD 텍스트 분석

        Args:
            jd_text: 추출된 JD 텍스트
            pdf_filename: PDF 파일명

        Returns:
            분석 결과 딕셔너리
        """
        prompt = f"""
다음은 채용 공고(JD) 문서입니다. 이 문서를 분석하여 다음 정보를 JSON 형식으로 추출해주세요:

1. company_name: 회사명 (명시되지 않았으면 PDF 파일명에서 추정)
2. job_title: 채용 직무명
3. key_skills: 필요한 핵심 역량 (리스트, 최대 5개)
4. culture_summary: 기업 문화 또는 인재상 요약 (50자 이내)

PDF 파일명: {pdf_filename}

채용 공고 내용:
{jd_text[:3000]}

반드시 다음 JSON 형식으로만 응답하세요:
{{
    "company_name": "회사명",
    "job_title": "직무명",
    "key_skills": ["역량1", "역량2", "역량3"],
    "culture_summary": "기업 문화 요약"
}}
"""

        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            })

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read().decode('utf-8'))
            llm_output = response_body['content'][0]['text'].strip()

            # JSON 파싱
            # LLM이 ```json ... ``` 형식으로 응답할 수 있으므로 처리
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].strip()

            result = json.loads(llm_output)
            print(f"Bedrock 분석 완료: {result['company_name']}")
            return result

        except Exception as e:
            print(f"Bedrock 분석 에러: {e}")
            # 기본값 반환
            return {
                "company_name": pdf_filename.replace(".pdf", ""),
                "job_title": "직무 미상",
                "key_skills": ["역량 미상"],
                "culture_summary": "기업 문화 정보 없음"
            }

    def parse_pdf_to_company_profile(self, pdf_path: str, company_id: str) -> CompanyProfile:
        """
        PDF를 CompanyProfile 객체로 변환

        Args:
            pdf_path: PDF 파일 경로
            company_id: 기업 ID

        Returns:
            CompanyProfile 객체
        """
        # PDF 텍스트 추출
        jd_text = self.extract_text_from_pdf(pdf_path)

        if not jd_text:
            raise ValueError(f"PDF에서 텍스트를 추출할 수 없습니다: {pdf_path}")

        # Bedrock으로 분석
        pdf_filename = os.path.basename(pdf_path)
        analysis = self.analyze_jd_with_bedrock(jd_text, pdf_filename)

        # CompanyProfile 생성
        profile = CompanyProfile(
            company_id=company_id,
            source_pdf=pdf_filename,
            company_name=analysis["company_name"],
            job_title=analysis["job_title"],
            key_skills=analysis["key_skills"],
            culture_summary=analysis["culture_summary"],
            job_description=jd_text[:500]  # 처음 500자만 저장
        )

        return profile

    def parse_all_pdfs(self, pdf_dir: str) -> List[CompanyProfile]:
        """
        디렉토리 내 모든 PDF 파싱

        Args:
            pdf_dir: PDF 파일이 있는 디렉토리

        Returns:
            CompanyProfile 리스트
        """
        profiles = []

        # PDF 파일 찾기
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        print(f"\n발견된 PDF 파일: {len(pdf_files)}개")

        for idx, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            company_id = f"company_{idx}"

            print(f"\n[{idx}/{len(pdf_files)}] 파싱 중: {pdf_file}")

            try:
                profile = self.parse_pdf_to_company_profile(pdf_path, company_id)
                profiles.append(profile)
                print(f"✓ 완료: {profile.company_name}")

            except Exception as e:
                print(f"✗ 에러: {e}")

        print(f"\n총 {len(profiles)}개 기업 프로필 생성 완료")
        return profiles
