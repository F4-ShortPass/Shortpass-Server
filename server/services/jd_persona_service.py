# server/services/jd_persona_service.py
"""
JD 기반 페르소나 관리 서비스
"""
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from models.jd_persona import JDPersona, JDPersonaQuestion
from services.competency_service import CompetencyService


class JDPersonaService:
    """
    JD 페르소나 생성 및 관리 서비스
    """

    def __init__(self):
        self.competency_service = CompetencyService()

    async def create_and_save_persona(
        self,
        db: Session,
        job_id: int,
        company_id: int,
        jd_text: str,
        company_questions: List[str]
    ) -> Dict[str, Any]:
        """
        JD로부터 페르소나 생성하고 DB에 저장

        Args:
            db: 데이터베이스 세션
            job_id: Job ID
            company_id: 회사 ID
            jd_text: JD 텍스트
            company_questions: 기업 질문 3개

        Returns:
            Dict: 생성된 페르소나 정보
        """
        try:
            print(f"🎭 Starting persona creation for Job {job_id}")

            # 1. 역량 분석
            competency_data = await self.competency_service.analyze_jd_competencies(jd_text)
            print(f" Extracted competencies: {len(competency_data['job_competencies'])} job-specific")

            # 2. 페르소나 생성
            persona_data = await self.competency_service.generate_persona_data(
                jd_text=jd_text,
                job_competencies=competency_data["job_competencies"],
                company_questions=company_questions
            )

            # 3. 시각화 데이터 생성
            visualization_data = self.competency_service.get_competency_visualization_data(
                job_competencies=competency_data["job_competencies"]
            )

            # 4. 전체 데이터 병합
            complete_persona_data = {
                **persona_data,
                "analysis_summary": competency_data.get("analysis_summary", "")
            }

            # 5. DB에 저장
            jd_persona = JDPersona.create_from_generation_result(
                job_id=job_id,
                company_id=company_id,
                generation_result=complete_persona_data,
                visualization_data=visualization_data
            )

            db.add(jd_persona)
            db.commit()
            db.refresh(jd_persona)

            print(f"✅ Persona saved to DB with ID: {jd_persona.id}")

            # 6. 응답 데이터 구성
            result = jd_persona.to_dict()
            result["visualization_data"] = visualization_data

            return result

        except Exception as e:
            db.rollback()
            print(f"❌ Failed to create persona: {e}")
            raise Exception(f"Failed to create persona: {str(e)}")

    def get_persona_by_job_id(
        self,
        db: Session,
        job_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Job ID로 페르소나 조회

        Args:
            db: 데이터베이스 세션
            job_id: Job ID

        Returns:
            Optional[Dict]: 페르소나 정보 또는 None
        """
        try:
            persona = db.query(JDPersona).filter(
                JDPersona.job_id == job_id,
                JDPersona.is_active == True
            ).first()

            if not persona:
                return None

            result = persona.to_dict()

            # 시각화 데이터 추가
            if persona.job_competencies:
                result["visualization_data"] = self.competency_service.get_competency_visualization_data(
                    persona.job_competencies
                )

            return result

        except Exception as e:
            print(f"❌ Failed to get persona: {e}")
            return None

    def update_persona_questions(
        self,
        db: Session,
        persona_id: int,
        new_questions: List[str]
    ) -> bool:
        """
        페르소나의 기업 질문 업데이트

        Args:
            db: 데이터베이스 세션
            persona_id: 페르소나 ID
            new_questions: 새로운 질문 리스트

        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            persona = db.query(JDPersona).filter(JDPersona.id == persona_id).first()

            if not persona:
                return False

            persona.core_questions = new_questions
            db.commit()

            print(f"✅ Updated questions for persona {persona_id}")
            return True

        except Exception as e:
            db.rollback()
            print(f"❌ Failed to update questions: {e}")
            return False

    def deactivate_persona(
        self,
        db: Session,
        persona_id: int
    ) -> bool:
        """
        페르소나 비활성화

        Args:
            db: 데이터베이스 세션
            persona_id: 페르소나 ID

        Returns:
            bool: 비활성화 성공 여부
        """
        try:
            persona = db.query(JDPersona).filter(JDPersona.id == persona_id).first()

            if not persona:
                return False

            persona.is_active = False
            db.commit()

            print(f"✅ Deactivated persona {persona_id}")
            return True

        except Exception as e:
            db.rollback()
            print(f"❌ Failed to deactivate persona: {e}")
            return False

    def get_company_personas(
        self,
        db: Session,
        company_id: int,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        회사의 모든 페르소나 조회

        Args:
            db: 데이터베이스 세션
            company_id: 회사 ID
            active_only: 활성화된 것만 조회할지 여부

        Returns:
            List[Dict]: 페르소나 리스트
        """
        try:
            query = db.query(JDPersona).filter(JDPersona.company_id == company_id)

            if active_only:
                query = query.filter(JDPersona.is_active == True)

            personas = query.order_by(JDPersona.created_at.desc()).all()

            return [persona.to_dict() for persona in personas]

        except Exception as e:
            print(f"❌ Failed to get company personas: {e}")
            return []

    async def regenerate_persona(
        self,
        db: Session,
        job_id: int,
        new_questions: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        페르소나 재생성

        Args:
            db: 데이터베이스 세션
            job_id: Job ID
            new_questions: 새로운 질문들 (선택사항)

        Returns:
            Optional[Dict]: 재생성된 페르소나 정보
        """
        try:
            # 기존 페르소나 비활성화
            existing = db.query(JDPersona).filter(
                JDPersona.job_id == job_id,
                JDPersona.is_active == True
            ).first()

            if existing:
                existing.is_active = False

            # Job 정보 조회 (실제 구현에서는 JobService 사용)
            # 임시로 기존 데이터 사용
            if not new_questions:
                new_questions = existing.core_questions if existing else [
                    "프로젝트 경험을 설명해주세요.",
                    "팀워크 경험을 말해주세요.",
                    "도전적인 과제를 어떻게 해결했나요?"
                ]

            # 새로운 페르소나 생성
            # 실제로는 JD 텍스트를 다시 가져와야 함
            # 여기서는 생략하고 기존 정보 재사용
            company_id = existing.company_id if existing else 1

            # 새 페르소나 생성 (실제 구현에서는 complete flow 필요)
            # 임시로 기존 로직 재사용
            print(f"🔄 Regenerating persona for Job {job_id}")

            return {"message": "Persona regeneration requires full JD text - implement in next iteration"}

        except Exception as e:
            db.rollback()
            print(f"❌ Failed to regenerate persona: {e}")
            return None