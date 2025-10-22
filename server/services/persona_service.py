# server/services/persona_service.py
"""
페르소나 서비스
"""
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from models.interview import PersonaDB, Question, Company
from schemas.persona import PersonaCreate, QuestionCreate
from services.s3_service import S3Service
from ai.parsers.persona_question_parser import PersonaQuestionParser
from services.persona_generator import PersonaGenerator
from models.company_profile import CompanyProfile
from models.persona import ArchetypeEnum


class PersonaService:
    """페르소나 관련 비즈니스 로직"""

    def __init__(self, db: Session):
        self.db = db
        self.s3_service = S3Service()
        self.parser = PersonaQuestionParser()
        self.generator = PersonaGenerator()

   
    def create_persona_from_pdf(
        self,
        company_id: int,
        pdf_file_content: bytes,
        pdf_file_name: str
    ) -> Dict[str, Any]:
        """
        PDF 파일로부터 페르소나 생성

        Args:
            company_id: 회사 ID
            pdf_file_content: PDF 파일 내용
            pdf_file_name: PDF 파일 이름

        Returns:
            Dict containing persona and questions
        """
        # 1. 회사 정보 조회
        company = self.db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise ValueError(f"Company with id {company_id} not found")

        company_name = company.name

        # 2. PDF 업로드
        print(f"\n📤 PDF 업로드 중: {pdf_file_name}")
        s3_path = self.s3_service.upload_file(
            file_content=pdf_file_content,
            file_name=pdf_file_name,
            folder=f"personas/company_{company_id}"
        )
        print(f"✓ S3 업로드 완료: {s3_path}")

        # 3. PDF 파싱 (질문 추출)
        parsed_data = self.parser.parse_persona_questions(pdf_file_content, company_name)

        persona_info = parsed_data["persona_info"]
        questions_data = parsed_data["questions"]

        # 4. 페르소나 생성 (PersonaGenerator 사용)
        # CompanyProfile 생성 (persona_info에서 정보 추출)
        company_profile = CompanyProfile(
            company_id=str(company_id),
            company_name=company_name,
            job_title=persona_info.get("persona_name", "면접관"),
            key_skills=persona_info.get("focus_areas", []),
            culture_summary=persona_info.get("description", ""),
            source_pdf=pdf_file_name,
            job_description=parsed_data.get("full_text", "")
        )

        # Archetype 매핑
        archetype_map = {
            "analytical": ArchetypeEnum.ANALYTICAL,
            "supportive": ArchetypeEnum.SUPPORTIVE,
            "stress_tester": ArchetypeEnum.STRESS_TESTER
        }
        archetype_str = persona_info.get("archetype", "analytical")
        archetype = archetype_map.get(archetype_str, ArchetypeEnum.ANALYTICAL)

        # PersonaGenerator로 system_prompt와 welcome_message 생성
        persona_obj = self.generator.create_persona(
            company_profile=company_profile,
            archetype=archetype,
            persona_id=f"persona_{company_id}"
        )

        # 5. DB에 페르소나 저장
        persona_db = PersonaDB(
            company_id=company_id,
            persona_name=persona_info.get("persona_name", f"{company_name} 면접관"),
            archetype=archetype_str,
            description=persona_info.get("description", ""),
            system_prompt=persona_obj.system_prompt,
            welcome_message=persona_obj.welcome_message,
            style_description=persona_obj.style_description,
            focus_keywords=persona_obj.focus_keywords,
            focus_areas=persona_info.get("focus_areas", []),
            pdf_file_path=s3_path,
            parsed_data=parsed_data
        )

        self.db.add(persona_db)
        self.db.commit()
        self.db.refresh(persona_db)

        print(f"✓ 페르소나 DB 저장 완료: ID {persona_db.id}")

        # 6. 질문들 저장
        saved_questions = []
        for q_data in questions_data:
            question = Question(
                persona_id=persona_db.id,
                question_type=q_data.get("question_type", "general"),
                question_text=q_data["question_text"],
                expected_keywords=q_data.get("expected_keywords", []),
                evaluation_criteria=q_data.get("evaluation_criteria", []),
                difficulty_level=q_data.get("difficulty_level", 3)
            )
            self.db.add(question)
            saved_questions.append(question)

        self.db.commit()

        # 질문들 refresh
        for q in saved_questions:
            self.db.refresh(q)

        print(f"✓ {len(saved_questions)}개 질문 DB 저장 완료")

        return {
            "persona": persona_db,
            "questions": saved_questions
        }

    def get_persona(self, persona_id: int) -> Optional[PersonaDB]:
        """페르소나 조회"""
        return self.db.query(PersonaDB).filter(PersonaDB.id == persona_id).first()

    def get_personas_by_company(self, company_id: int) -> List[PersonaDB]:
        """회사별 페르소나 목록 조회"""
        return self.db.query(PersonaDB).filter(PersonaDB.company_id == company_id).all()

    def get_all_personas(self) -> List[PersonaDB]:
        """전체 페르소나 목록 조회"""
        return self.db.query(PersonaDB).all()

    def get_persona_questions(self, persona_id: int) -> List[Question]:
        """페르소나의 질문 목록 조회"""
        return self.db.query(Question).filter(Question.persona_id == persona_id).all()

    def delete_persona(self, persona_id: int) -> bool:
        """페르소나 삭제"""
        persona = self.get_persona(persona_id)
        if not persona:
            return False

        # S3에서 파일 삭제
        if persona.pdf_file_path:
            try:
                self.s3_service.delete_file(persona.pdf_file_path)
            except Exception as e:
                print(f"⚠️  S3 파일 삭제 실패: {e}")

        # 연관된 질문들도 삭제됨 (cascade)
        self.db.delete(persona)
        self.db.commit()

        print(f"✓ 페르소나 삭제 완료: ID {persona_id}")
        return True
