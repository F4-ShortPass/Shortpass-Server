#!/usr/bin/env python3
"""
Mock 데이터 생성 스크립트
- Company, Job, JDPersona, PersonaInstance 생성
"""
import sys
import json
from pathlib import Path
from sqlalchemy.orm import Session

# 경로 설정
sys.path.append(str(Path(__file__).parent))

from db.database import SessionLocal, engine, Base
from models.company import Company
from models.job import Job
from models.jd_persona import JDPersona
from models.interview import PersonaInstance

# 모든 테이블 생성
Base.metadata.create_all(bind=engine)

def load_persona_data():
    """persona_data.json 로드"""
    persona_file = Path(__file__).parent / "assets" / "persona_data.json"
    with open(persona_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_mock_data():
    """Mock 데이터 생성"""
    db = SessionLocal()

    try:
        print("\n" + "="*60)
        print("Mock 데이터 생성 시작")
        print("="*60)

        # persona_data.json 로드
        persona_data = load_persona_data()

        # 1. Company 생성/조회
        company = db.query(Company).filter(Company.id == 1).first()
        if not company:
            company = Company(
                id=1,
                name="삼성물산 패션부문",
                size="대기업",
                values="혁신, 도전, 글로벌",
                blind=False
            )
            db.add(company)
            db.flush()
            print(f"✅ Company 생성: {company.name} (ID: {company.id})")
        else:
            print(f"✅ Company 이미 존재: {company.name} (ID: {company.id})")

        # 2. Job 생성/조회
        job = db.query(Job).filter(Job.id == 1).first()
        if not job:
            # 직무 역량 5개 (persona_data의 core_competencies에서 추출)
            job_competencies_5 = [
                "Market & Trend Insight",
                "Strategic Thinking",
                "Creativity & Execution",
                "Communication",
                "Global & Business Mindset"
            ]

            job = Job(
                id=1,
                company_id=company.id,
                title="상품기획(MD/MR) / Retail영업",
                description=persona_data.get("job_info", {}).get("description_summary", ""),
                dynamic_evaluation_criteria=json.dumps(job_competencies_5),
                position_type="MD/MR",
                seniority_level="Junior-Senior"
            )
            db.add(job)
            db.flush()
            print(f"✅ Job 생성: {job.title} (ID: {job.id})")
        else:
            print(f"✅ Job 이미 존재: {job.title} (ID: {job.id})")

        # 3. JDPersona 생성/조회
        jd_persona = db.query(JDPersona).filter(JDPersona.job_id == job.id).first()

        # 공통 역량 5개 (기존 6개에서 5개로 축소)
        common_competencies_5 = [
            "고객지향",
            "도전정신",
            "협동",
            "목표지향",
            "책임감"
        ]

        # 직무 역량 5개
        job_competencies_5 = [
            "Market & Trend Insight",
            "Strategic Thinking",
            "Creativity & Execution",
            "Communication",
            "Global & Business Mindset"
        ]

        # 초기 질문 (persona_data에서)
        initial_questions = persona_data.get("initial_questions", [
            "삼성물산 패션부문에 지원해주셔서 감사합니다. 간단히 자기소개 부탁드립니다.",
            "최근 가장 인상 깊게 본 패션 트렌드는 무엇인가요?",
            "의견 차이가 발생했을 때 어떻게 해결하셨나요?"
        ])

        # 페르소나 요약 (3명의 면접관)
        persona_summary = [
            {
                "type": "전략적 사고형 면접관",
                "focus": "시장 분석 및 데이터 기반 의사결정 능력 평가",
                "target_competencies": ["Market & Trend Insight", "Strategic Thinking"],
                "style": "논리적이고 분석적, 구체적인 근거를 요구"
            },
            {
                "type": "실행력 중심형 면접관",
                "focus": "목표 달성을 위한 창의적 실행과 협업 능력 평가",
                "target_competencies": ["Creativity & Execution", "Communication"],
                "style": "실무 경험과 구체적 성과를 중시"
            },
            {
                "type": "글로벌 비즈니스형 면접관",
                "focus": "글로벌 감각과 비즈니스 마인드 평가",
                "target_competencies": ["Global & Business Mindset"],
                "style": "전략적 사고와 글로벌 시각을 평가"
            }
        ]

        if not jd_persona:
            jd_persona = JDPersona(
                job_id=job.id,
                company_id=company.id,
                company_name=company.name,
                common_competencies=common_competencies_5,
                job_competencies=job_competencies_5,
                core_questions=initial_questions,
                persona_summary=persona_summary,
                analysis_summary="삼성물산 패션부문 MD/영업 직무에 필요한 핵심 역량 분석 완료",
                is_active=True
            )
            db.add(jd_persona)
            db.flush()
            print(f"✅ JDPersona 생성 (ID: {jd_persona.id})")
            print(f"   - 공통 역량: {len(common_competencies_5)}개")
            print(f"   - 직무 역량: {len(job_competencies_5)}개")
            print(f"   - 페르소나: {len(persona_summary)}명")
        else:
            print(f"✅ JDPersona 이미 존재 (ID: {jd_persona.id})")

        # 4. PersonaInstance 생성 (3명의 면접관)
        for idx, persona_info in enumerate(persona_summary):
            persona_instance = db.query(PersonaInstance).filter(
                PersonaInstance.company_id == company.id,
                PersonaInstance.instance_name == persona_info["type"]
            ).first()

            if not persona_instance:
                persona_instance = PersonaInstance(
                    company_id=company.id,
                    instance_name=persona_info["type"],
                    system_prompt=persona_data.get("system_prompt", "당신은 전문 면접관입니다."),
                    focus_area=persona_info["focus"],
                    question_style=persona_info["style"],
                    target_competencies=json.dumps(persona_info["target_competencies"]),
                    is_active=True
                )
                db.add(persona_instance)
                db.flush()
                print(f"✅ PersonaInstance 생성: {persona_instance.instance_name} (ID: {persona_instance.id})")
            else:
                print(f"✅ PersonaInstance 이미 존재: {persona_instance.instance_name} (ID: {persona_instance.id})")

        # 커밋
        db.commit()

        print("\n" + "="*60)
        print("✅ Mock 데이터 생성 완료!")
        print("="*60)
        print(f"\n 생성된 데이터:")
        print(f"   - Company ID: {company.id}")
        print(f"   - Job ID: {job.id}")
        print(f"   - JDPersona ID: {jd_persona.id}")
        print(f"   - PersonaInstance: 3명")
        print(f"\n🎯 이제 프론트엔드에서 다음 ID들을 사용하세요:")
        print(f"   - companyId: {company.id}")
        print(f"   - jobId: {job.id}")

    except Exception as e:
        db.rollback()
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    setup_mock_data()
