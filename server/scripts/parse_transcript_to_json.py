"""
scripts.txt를 JSON 형식으로 파싱하는 스크립트

입력: server/docs/scripts.txt (텍스트 면접 transcript)
출력: server/test_data/transcript_박서연.json (구조화된 JSON)
"""

import json
import re
from typing import Dict, List, Any
from datetime import datetime


def parse_transcript(file_path: str) -> Dict[str, Any]:
    """
    텍스트 형식의 면접 transcript를 JSON으로 변환

    Args:
        file_path: scripts.txt 파일 경로

    Returns:
        구조화된 면접 데이터
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 메타데이터 추출
    company_match = re.search(r'기업:\s*(.+)', content)
    applicant_match = re.search(r'지원자:\s*(.+)', content)
    date_match = re.search(r'면접 일시:\s*(.+)', content)

    company = company_match.group(1).strip() if company_match else "Unknown"
    applicant_name = applicant_match.group(1).strip() if applicant_match else "Unknown"
    interview_date = date_match.group(1).strip() if date_match else "Unknown"

    # 질문-답변 쌍 추출
    qa_pattern = r'\[Question (\d+)\] 면접관: (.+?)\n\n면접관: (.+?)\n\n지원자: (.+?)(?=\n\n(?:\[Question|면접 종료))'

    qa_pairs = []
    matches = re.finditer(qa_pattern, content, re.DOTALL)

    for match in matches:
        question_num = int(match.group(1))
        evaluator_type = match.group(2).strip()
        question_text = match.group(3).strip()
        answer_text = match.group(4).strip()

        # 역량 매핑 (evaluator type → competency)
        competency_map = {
            "Strategic Thinking Evaluator": "strategic_thinking",
            "전략적 사고 평가자": "strategic_thinking",
            "Data-Driven Decision Evaluator": "data_driven",
            "데이터 기반 의사결정 평가자": "data_driven",
            "Communication & Collaboration Evaluator": "communication",
            "커뮤니케이션 및 협업 평가자": "communication",
            "Problem-Solving Evaluator": "problem_solving",
            "문제해결 능력 평가자": "problem_solving",
            "Automotive Industry Knowledge Evaluator": "industry_knowledge",
            "자동차 산업 이해도 평가자": "industry_knowledge",
            "Learning Attitude & Growth Potential Evaluator": "learning_attitude",
            "학습 태도 및 성장 가능성 평가자": "learning_attitude"
        }

        target_competencies = [competency_map.get(evaluator_type, "general")]

        qa_pairs.append({
            "id": f"q{question_num}",
            "question_number": question_num,
            "question_text": question_text,
            "answer_text": answer_text,
            "interviewer": evaluator_type,
            "target_competencies": target_competencies,
            "timestamp": f"00:{(question_num-1)*5:02d}:00",  # 가상 타임스탬프 (5분 간격)
            "duration_seconds": 300  # 5분
        })

    # 전체 구조 생성
    transcript_data = {
        "metadata": {
            "company": company,
            "applicant_name": applicant_name,
            "interview_date": interview_date,
            "total_questions": len(qa_pairs),
            "total_duration_minutes": len(qa_pairs) * 5,
            "interview_type": "AI-powered multi-persona"
        },
        "applicant_info": {
            "name": applicant_name,
            "education": "대졸 (경영학)",  # 추정
            "experience_years": 1,  # 인턴 경험 기반 추정
            "applied_position": company.split()[0] + " 전략기획팀"
        },
        "qa_pairs": qa_pairs,
        "personas": [
            {
                "name": "전략적 사고 평가자",
                "archetype": "strategic",
                "focus_competencies": ["strategic_thinking"],
                "question_count": sum(1 for qa in qa_pairs if "strategic_thinking" in qa["target_competencies"])
            },
            {
                "name": "데이터 기반 의사결정 평가자",
                "archetype": "analytical",
                "focus_competencies": ["data_driven"],
                "question_count": sum(1 for qa in qa_pairs if "data_driven" in qa["target_competencies"])
            },
            {
                "name": "커뮤니케이션 및 협업 평가자",
                "archetype": "collaborative",
                "focus_competencies": ["communication"],
                "question_count": sum(1 for qa in qa_pairs if "communication" in qa["target_competencies"])
            },
            {
                "name": "문제해결 능력 평가자",
                "archetype": "problem_solver",
                "focus_competencies": ["problem_solving"],
                "question_count": sum(1 for qa in qa_pairs if "problem_solving" in qa["target_competencies"])
            },
            {
                "name": "산업 이해도 평가자",
                "archetype": "domain_expert",
                "focus_competencies": ["industry_knowledge"],
                "question_count": sum(1 for qa in qa_pairs if "industry_knowledge" in qa["target_competencies"])
            },
            {
                "name": "학습 태도 평가자",
                "archetype": "growth_focused",
                "focus_competencies": ["learning_attitude"],
                "question_count": sum(1 for qa in qa_pairs if "learning_attitude" in qa["target_competencies"])
            }
        ]
    }

    return transcript_data


def main():
    """메인 실행 함수"""
    import os

    # 경로 설정
    input_file = "/home/ec2-user/flex/server/docs/scripts.txt"
    output_dir = "/home/ec2-user/flex/server/test_data"
    output_file = os.path.join(output_dir, "transcript_박서연.json")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 파싱 실행
    print(f"📄 Parsing transcript from: {input_file}")
    transcript_data = parse_transcript(input_file)

    # JSON 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Transcript saved to: {output_file}")
    print(f"\n Summary:")
    print(f"   - Applicant: {transcript_data['metadata']['applicant_name']}")
    print(f"   - Company: {transcript_data['metadata']['company']}")
    print(f"   - Total Questions: {transcript_data['metadata']['total_questions']}")
    print(f"   - Total Duration: {transcript_data['metadata']['total_duration_minutes']} minutes")

    print(f"\n👥 Personas:")
    for persona in transcript_data['personas']:
        print(f"   - {persona['name']}: {persona['question_count']} questions")

    print(f"\n💬 Sample QA:")
    sample_qa = transcript_data['qa_pairs'][0]
    print(f"   Q: {sample_qa['question_text'][:80]}...")
    print(f"   A: {sample_qa['answer_text'][:80]}...")


if __name__ == "__main__":
    main()
