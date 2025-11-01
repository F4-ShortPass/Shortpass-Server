"""
Dynamic job competency prompt builder.

Goal:
- JD 파싱 결과(job_competencies)를 받아 5개의 직무 역량 에이전트가
  공통 템플릿을 공유하면서, 각 역량에 특화된 프롬프트를 사용할 수 있게 생성.
- 여기서는 프롬프트만 생성하며, 실행/연결은 하지 않는다.
- 정성적 지시를 강화해, JD 맥락을 반영한 “왜 이 역량이 중요한지”를 설명하도록 유도.
"""

from typing import Dict, List, Optional

BASE_TEMPLATE = """You are an HR evaluator assessing one job-specific competency for a retail/fashion MD role.
Explain *why* this competency matters for the role, using the JD context, before you score.

Input:
- Competency name: {competency_name}
- Competency description (from JD): {competency_description}
- JD summary (optional): {jd_summary}
- Candidate transcript: {{transcript_content}}

Task:
1) Briefly restate (2-3문장) why this competency is critical for this JD (정성적 맥락 강조).
2) Assess only this competency. Do not grade unrelated skills.
3) Extract 3-6 positive evidences and 2-4 risks/weaknesses from the transcript.
4) Score 0-100 and a confidence_v2 0.0-1.0. Keep numbers but avoid revealing them in prose later.
5) Provide structured fields:
   - overall_score (0-100)
   - confidence_v2 (0.0-1.0)
   - strengths: list of 1-sentence bullet strings
   - weaknesses: list of 1-sentence bullet strings
   - key_observations: 3-5 short findings (no raw scores)
   - perspectives.evidence_details: list of evidences with segment_id/char_index/char_length if available
   - competency_rationale: 2-3 sentences on why this competency is important for this JD (정성 요약)
6) Keep tone concise, professional, and in Korean. Do not include the raw prompt in output.

Constraints:
- Do not fabricate segment_id/char_index/char_length; if unknown, set them to null/0.
- Do not copy-paste JD; use it only to calibrate what “good” looks like.
- Focus on behavioral proof from the transcript, not generic claims.
- Reason about *fit* for this JD, not generic job competence.
"""


def build_job_competency_prompts(
    job_competencies: List[Dict[str, str]],
    jd_summary: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build per-competency prompts for job-specific agents.

    Args:
        job_competencies: [{"name": "...", "description": "..."}] parsed from JD.
        jd_summary: Optional short JD summary to give additional context.

    Returns:
        dict mapping competency_name -> prompt string
    """
    prompts: Dict[str, str] = {}
    jd_text = jd_summary or "N/A"

    for comp in job_competencies:
        name = comp.get("name") or comp.get("competency_name") or "unknown_competency"
        desc = comp.get("description") or comp.get("details") or "No description provided."

        prompts[name] = BASE_TEMPLATE.format(
            competency_name=name,
            competency_description=desc,
            jd_summary=jd_text,
        )

    return prompts


__all__ = ["build_job_competency_prompts", "BASE_TEMPLATE"]
