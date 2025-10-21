"""
기업 프로필 데이터 모델
PDF에서 추출한 기업 정보를 저장합니다.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CompanyProfile:
    """기업 프로필"""

    company_id: str                    # 기업 ID (예: "company_1")
    source_pdf: str                     # PDF 파일명
    company_name: str                   # 회사명 (추정)
    job_title: str                      # 채용 직무
    key_skills: List[str]               # 필요 역량
    culture_summary: str                # 기업 문화/인재상 요약
    job_description: str                # 직무 설명 (원문)

    def __str__(self):
        return f"CompanyProfile({self.company_name} - {self.job_title})"

    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            "company_id": self.company_id,
            "source_pdf": self.source_pdf,
            "company_name": self.company_name,
            "job_title": self.job_title,
            "key_skills": self.key_skills,
            "culture_summary": self.culture_summary,
            "job_description": self.job_description
        }
