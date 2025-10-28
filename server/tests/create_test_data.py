#!/usr/bin/env python3
# server/create_test_data.py
"""
Create test data for interview testing.
"""
from db.database import SessionLocal
from models.interview import Applicant, Company, PersonaDB

db = SessionLocal()

try:
    # Create test companies (ID 명시하지 않음 - 자동 증가)
    company1 = Company(
        name="테스트 회사 1",
        company_values_text="혁신과 도전을 중시합니다",
        blind_mode=False
    )
    company2 = Company(
        name="테스트 회사 2",
        company_values_text="팀워크와 협업을 중시합니다",
        blind_mode=False
    )
    db.add(company1)
    db.add(company2)
    db.flush()  # ID 할당을 위해 flush

    # Create test personas
    persona1 = PersonaDB(
        company_id=company1.id,
        persona_name="기술 면접관",
        archetype="analytical",
        description="기술적 깊이를 중시하는 면접관",
        system_prompt="당신은 기술적 역량을 평가하는 면접관입니다.",
        welcome_message="안녕하세요. 기술 면접을 시작하겠습니다.",
        focus_keywords=["Python", "알고리즘", "시스템 설계"],
        focus_areas=["기술 스킬", "문제 해결"]
    )
    persona2 = PersonaDB(
        company_id=company1.id,
        persona_name="문화 적합도 면접관",
        archetype="supportive",
        description="조직 문화 적합도를 평가하는 면접관",
        system_prompt="당신은 문화 적합도를 평가하는 면접관입니다.",
        welcome_message="안녕하세요. 문화 적합도 면접을 시작하겠습니다.",
        focus_keywords=["팀워크", "커뮤니케이션", "협업"],
        focus_areas=["소프트 스킬", "문화 적합도"]
    )
    persona3 = PersonaDB(
        company_id=company2.id,
        persona_name="경험 평가 면접관",
        archetype="analytical",
        description="실무 경험을 평가하는 면접관",
        system_prompt="당신은 실무 경험을 평가하는 면접관입니다.",
        welcome_message="안녕하세요. 경험 평가 면접을 시작하겠습니다.",
        focus_keywords=["프로젝트", "경력", "성과"],
        focus_areas=["실무 경험", "성과"]
    )
    db.add(persona1)
    db.add(persona2)
    db.add(persona3)

    db.commit()
    print("✓ Test data created successfully")
    print(f"  - Companies: {company1.name} (ID: {company1.id}), {company2.name} (ID: {company2.id})")
    print(f"  - Personas: 3 personas created")
    print("\nNote: No test applicant created to avoid ID conflicts with real users")

except Exception as e:
    print(f"✗ Error creating test data: {e}")
    db.rollback()
finally:
    db.close()
