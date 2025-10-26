"""
Evidence Conflict Detector (Phase 2)
같은 segment를 여러 역량 Agent가 다르게 해석한 경우 탐지
LLM 호출 없이 규칙 기반으로 빠르게 처리

충돌 기준:
- 같은 segment_id를 2개 이상 역량이 evidence_details에서 참조
- quality_score 차이가 0.4 이상 (40점 이상)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class EvidenceConflict:
    """충돌 정보"""
    segment_id: str
    competencies: List[str]  # 충돌된 역량들
    scores: List[float]      # 각 역량의 quality_score (0.0-1.0)
    gap: float               # 최대 점수 차이 (0.0-1.0)
    priority: int            # 충돌 심각도 순위 (1=가장 심각)
    
    def to_dict(self) -> Dict:
        """Dict 변환"""
        return {
            "segment_id": self.segment_id,
            "competencies": self.competencies,
            "scores": self.scores,
            "gap": round(self.gap, 2),
            "priority": self.priority
        }


class ConflictDetector:
    """Evidence 충돌 탐지기"""
    
    @staticmethod
    def detect_conflicts(
        all_results: Dict[str, Dict],
        threshold: float = 0.4,
        max_conflicts: int = 5
    ) -> List[EvidenceConflict]:
        """
        모든 역량 평가 결과에서 Evidence 충돌 탐지
        
        Args:
            all_results: 10개 역량 평가 결과
            threshold: 점수 차이 임계값 (기본 0.4 = 40점)
            max_conflicts: 최대 반환 개수 (Phase 3 비용 제한)
        
        Returns:
            충돌 목록 (심각도 순 정렬)
        """
        # 1. Segment별로 어떤 역량들이 참조했는지 매핑
        segment_map = ConflictDetector._build_segment_map(all_results)
        
        # 2. 각 Segment에서 점수 차이 확인
        conflicts = ConflictDetector._find_conflicts(segment_map, threshold)
        
        # 3. 충돌 심각도 순으로 정렬 (gap 큰 순서)
        conflicts_sorted = sorted(conflicts, key=lambda c: c.gap, reverse=True)
        
        # 4. Priority 부여
        for idx, conflict in enumerate(conflicts_sorted, start=1):
            conflict.priority = idx
        
        # 5. 최대 개수 제한
        return conflicts_sorted[:max_conflicts]
    
    @staticmethod
    def _build_segment_map(all_results: Dict[str, Dict]) -> Dict[str, List[Tuple[str, float]]]:
        """Segment별 참조 매핑 구축"""
        segment_map = {}
        
        for comp_name, result in all_results.items():
            if "error" in result:
                continue
            
            perspectives = result.get("perspectives", {})
            evidence_details = perspectives.get("evidence_details", [])
            
            for evidence in evidence_details:
                segment_id = str(evidence.get("segment_id", ""))
                quality_score = evidence.get("quality_score", 0.0)
                
                if segment_id:
                    if segment_id not in segment_map:
                        segment_map[segment_id] = []
                    segment_map[segment_id].append((comp_name, quality_score))
        
        return segment_map
    
    @staticmethod
    def _find_conflicts(
        segment_map: Dict[str, List[Tuple[str, float]]],
        threshold: float
    ) -> List[EvidenceConflict]:
        """Segment별 점수 차이 확인하여 충돌 찾기"""
        conflicts = []
        
        for segment_id, references in segment_map.items():
            if len(references) < 2:
                continue
            
            competencies = [comp for comp, _ in references]
            scores = [score for _, score in references]
            
            max_score = max(scores)
            min_score = min(scores)
            gap = max_score - min_score
            
            if gap >= threshold:
                conflicts.append(EvidenceConflict(
                    segment_id=segment_id,
                    competencies=competencies,
                    scores=scores,
                    gap=gap,
                    priority=0
                ))
        
        return conflicts
    
    @staticmethod
    def format_conflicts_for_log(conflicts: List[EvidenceConflict]) -> str:
        """충돌 정보를 로그용 문자열로 포맷"""
        if not conflicts:
            return "충돌 없음"
        
        lines = [f"총 {len(conflicts)}건의 충돌 탐지:"]
        for conflict in conflicts:
            comp_scores = ", ".join([
                f"{comp}={score:.2f}"
                for comp, score in zip(conflict.competencies, conflict.scores)
            ])
            lines.append(
                f"  Priority {conflict.priority}. Segment {conflict.segment_id}: "
                f"{comp_scores} (gap: {conflict.gap:.2f})"
            )
        
        return "\n".join(lines)
    
    @staticmethod
    def get_conflict_summary(conflicts: List[EvidenceConflict]) -> Dict:
        """충돌 요약 정보"""
        if not conflicts:
            return {
                "total_conflicts": 0,
                "max_gap": 0.0,
                "affected_segments": [],
                "affected_competencies": []
            }
        
        affected_segments = list(set(c.segment_id for c in conflicts))
        affected_competencies = list(set(
            comp for c in conflicts for comp in c.competencies
        ))
        
        return {
            "total_conflicts": len(conflicts),
            "max_gap": round(max(c.gap for c in conflicts), 2),
            "affected_segments": sorted(affected_segments),
            "affected_competencies": sorted(affected_competencies)
        }