"""
Base Agent
모든 역량 평가 Agent의 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import json
from datetime import datetime
from services.storage.s3_service import S3Service
from core import config # Import config

class BaseAgent(ABC):
    """역량 평가 Agent 기본 클래스"""
    
    def __init__(
        self, 
        bedrock_client,
        competency_name: str,
        competency_display_name: str,
        competency_category: str
    ):
        """
        Args:
            bedrock_client: AWS Bedrock 클라이언트
            competency_name: 역량 식별자 (예: "problem_solving")
            competency_display_name: 역량 표시명 (예: "문제해결력")
            competency_category: "job" or "common"
        """
        self.bedrock_client = bedrock_client
        self.competency_name = competency_name
        self.competency_display_name = competency_display_name
        self.competency_category = competency_category
        self.s3_service = S3Service(bucket_name=config.S3_BUCKET_NAME, region_name=config.AWS_REGION) # Initialize S3Service
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        역량별 프롬프트 템플릿 반환
        
        Returns:
            프롬프트 문자열
        """
        pass
    
    def prepare_transcript(self, transcript: Dict) -> str:
        """
        Transcript를 프롬프트에 넣을 형태로 변환
        
        Args:
            transcript: Transcript JSON
        
        Returns:
            JSON 문자열
        """
        return json.dumps(transcript, ensure_ascii=False, indent=2)
    
    async def evaluate(
        self, 
        transcript: Dict
    ) -> Dict:
        """
        역량 평가 실행
        
        Args:
            transcript: Interview Transcript JSON
        
        Returns:
            CompetencyEvaluation Dict
        """
        try:
            # 1. 프롬프트 생성
            prompt_template = self.get_prompt_template()
            transcript_str = self.prepare_transcript(transcript)
            
            prompt = prompt_template.format(
                transcript=transcript_str,
                resume=""  # Resume는 사용 안 함
            )
            
            # 2. Bedrock API 호출
            response = await self.bedrock_client.invoke_model(
                model_id="anthropic.claude-sonnet-4-20250514",
                prompt=prompt,
                temperature=0.0,  # 결정성 확보
                max_tokens=4000
            )
            
            # 3. JSON 파싱
            result = json.loads(response)
            
            # 4. 기본 정보 추가
            result["competency_name"] = self.competency_name
            result["competency_display_name"] = self.competency_display_name
            result["competency_category"] = self.competency_category
            result["evaluated_at"] = datetime.now().isoformat()
            
            # Agent 실행 로그 S3에 저장
            log_id = f"{self.competency_name}-{result['evaluated_at']}"
            self.s3_service.save_agent_log(result, log_id)
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"Evaluation failed for {self.competency_name}: {e}")
    
    def extract_relevant_segments(
        self, 
        transcript: Dict, 
        target_competencies: Optional[list] = None
    ) -> Dict:
        """
        특정 역량 관련 Segment만 추출 (선택적)
        
        Args:
            transcript: 전체 Transcript
            target_competencies: 필터링할 역량 목록
        
        Returns:
            필터링된 Transcript
        """
        if not target_competencies:
            # 필터링 없이 전체 반환
            return transcript
        
        segments = transcript.get("segments", [])
        filtered_segments = [
            seg for seg in segments
            if any(comp in seg.get("target_competencies", []) for comp in target_competencies)
        ]
        
        filtered_transcript = transcript.copy()
        filtered_transcript["segments"] = filtered_segments
        
        return filtered_transcript