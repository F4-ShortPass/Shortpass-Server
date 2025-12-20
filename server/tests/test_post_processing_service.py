import sys
from pathlib import Path

# Ensure server package importable when running pytest from repo root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from services.evaluation.post_processing_service import PostProcessingService


def test_post_processing_service_basic():
    service = PostProcessingService(
        positive_top_k=3, negative_top_k=2, question_top_k=2, low_score_threshold=80
    )
    aggregated = {
        "job_comp": {
            "overall_score": 78,
            "strengths": ["데이터 기반 문제 정의"],
            "weaknesses": ["리스크 관리 세부 부족"],
        },
        "common_comp": {
            "overall_score": 90,
            "strengths": ["이해관계자 설득"],
            "weaknesses": [],
        },
    }
    final_result = {"final_score": 85.0, "avg_confidence": 0.8, "overall_evaluation_summary": "테스트 요약"}

    summary = service.build_analysis_summary(aggregated, final_result)

    assert summary["aggregator_summary"] == "테스트 요약"
    assert summary["overall_applicant_summary"]
    assert summary["positive_keywords"], "positive keywords should not be empty"
    assert summary["negative_keywords"], "negative keywords should not be empty"
    assert summary["recommended_questions"], "recommended questions should not be empty"
