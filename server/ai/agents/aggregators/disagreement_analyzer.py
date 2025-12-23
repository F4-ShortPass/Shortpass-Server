"""
Disagreement Analyzer
Inter-agent variance detection for uncertainty quantification

References:
    - EMNLP 2024: LLM-TOPLA (Diversity-Driven Ensembles)
    - arXiv 2025: Multi-Agent Debate with Confidence Expression
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DisagreementResult:
    """Disagreement analysis result"""
    segment_id: int
    score_variance: float
    confidence_variance: float
    competencies_evaluated: int
    disagreement_score: float  # Normalized [0, 1]
    consensus_level: str  # "high"/"medium"/"low"
    requires_review: bool


class DisagreementAnalyzer:
    """
    Analyze inter-agent disagreement for uncertainty quantification

    Purpose:
        - Detect segments where multiple agents disagree on evaluation
        - Use disagreement as a signal for low confidence/uncertainty
        - Flag evaluations that need human review

    Strategy:
        - Compute variance in scores across agents
        - High variance → high disagreement → lower confidence
        - Low variance → high consensus → higher confidence
    """

    # Consensus thresholds
    HIGH_CONSENSUS_VARIANCE = 100.0  # Low variance threshold
    MEDIUM_CONSENSUS_VARIANCE = 400.0  # Medium variance threshold

    # Review threshold
    REVIEW_THRESHOLD = 0.65  # Disagreement score > 0.65 requires review


    @staticmethod
    def compute_segment_disagreement(
        segment_evaluations: List[Dict]
    ) -> Dict[int, DisagreementResult]:
        """
        Compute disagreement per segment across competencies

        Args:
            segment_evaluations: List of segment-level evaluations
                [
                    {
                        "competency": "achievement_motivation",
                        "segment_id": 3,
                        "score": 85,
                        "confidence_v2": 0.88
                    },
                    ...
                ]

        Returns:
            {
                segment_id: DisagreementResult
            }
        """
        # Group by segment_id
        segments_by_id: Dict[int, List[Dict]] = {}

        for eval_item in segment_evaluations:
            seg_id = eval_item.get("segment_id")
            if seg_id is None:
                continue

            if seg_id not in segments_by_id:
                segments_by_id[seg_id] = []

            segments_by_id[seg_id].append(eval_item)

        # Compute disagreement per segment
        disagreement_results = {}

        for seg_id, evals in segments_by_id.items():
            result = DisagreementAnalyzer._analyze_segment(seg_id, evals)
            disagreement_results[seg_id] = result

        return disagreement_results


    @staticmethod
    def _analyze_segment(
        segment_id: int,
        evaluations: List[Dict]
    ) -> DisagreementResult:
        """Analyze disagreement for a single segment"""

        if len(evaluations) < 2:
            # No disagreement with single evaluation
            return DisagreementResult(
                segment_id=segment_id,
                score_variance=0.0,
                confidence_variance=0.0,
                competencies_evaluated=len(evaluations),
                disagreement_score=0.0,
                consensus_level="high",
                requires_review=False
            )

        # Extract scores and confidences
        scores = [e.get("score", 0) for e in evaluations]
        confidences = [e.get("confidence_v2", 0.5) for e in evaluations]

        # Compute variance
        score_var = float(np.var(scores))
        conf_var = float(np.var(confidences))

        # Normalized disagreement (0-1 range)
        # Max variance for 0-100 scores with uniform distribution ≈ 833
        # We use 400 as practical threshold for "high disagreement"
        disagreement = min(1.0, score_var / 400.0)

        # Determine consensus level
        if score_var < DisagreementAnalyzer.HIGH_CONSENSUS_VARIANCE:
            consensus_level = "high"
        elif score_var < DisagreementAnalyzer.MEDIUM_CONSENSUS_VARIANCE:
            consensus_level = "medium"
        else:
            consensus_level = "low"

        # Flag for review
        requires_review = disagreement > DisagreementAnalyzer.REVIEW_THRESHOLD

        return DisagreementResult(
            segment_id=segment_id,
            score_variance=score_var,
            confidence_variance=conf_var,
            competencies_evaluated=len(evaluations),
            disagreement_score=disagreement,
            consensus_level=consensus_level,
            requires_review=requires_review
        )


    @staticmethod
    def compute_competency_disagreement(
        competency_results: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Compute disagreement across competencies

        Measures how much competency evaluations disagree with each other.
        Useful for detecting conflicting signals.

        Args:
            competency_results: 10 competency evaluation results
                {
                    "achievement_motivation": {"overall_score": 85, ...},
                    "growth_potential": {"overall_score": 72, ...},
                    ...
                }

        Returns:
            {
                "overall_variance": 95.3,
                "common_variance": 82.1,
                "job_variance": 108.5,
                "std_dev": 9.76
            }
        """
        all_scores = []
        common_scores = []
        job_scores = []

        common_competencies = [
            "achievement_motivation",
            "growth_potential",
            "interpersonal_skill",
            "organizational_fit",
            "problem_solving"
        ]

        for comp_name, comp_result in competency_results.items():
            score = comp_result.get("overall_score", 0)
            all_scores.append(score)

            if comp_name in common_competencies:
                common_scores.append(score)
            else:
                job_scores.append(score)

        overall_var = float(np.var(all_scores)) if all_scores else 0.0
        common_var = float(np.var(common_scores)) if common_scores else 0.0
        job_var = float(np.var(job_scores)) if job_scores else 0.0
        std_dev = float(np.std(all_scores)) if all_scores else 0.0

        return {
            "overall_variance": overall_var,
            "common_variance": common_var,
            "job_variance": job_var,
            "std_dev": std_dev,
            "score_range": max(all_scores) - min(all_scores) if all_scores else 0
        }


    @staticmethod
    def adjust_confidence_by_disagreement(
        confidence_v2: float,
        disagreement_score: float,
        penalty_weight: float = 0.3
    ) -> float:
        """
        Penalize confidence when agents disagree

        Strategy:
            - High disagreement → apply penalty to confidence
            - Low disagreement → no penalty (agents agree)

        Args:
            confidence_v2: Original confidence from Confidence V2
            disagreement_score: Disagreement metric (0-1)
            penalty_weight: How much to penalize (0-1, default 0.3)

        Returns:
            Adjusted confidence (clamped to [0.3, 0.98])

        Example:
            >>> conf = 0.85
            >>> disagreement = 0.7  # High disagreement
            >>> adjusted = adjust_confidence_by_disagreement(conf, disagreement)
            >>> print(adjusted)  # 0.67 (penalized)
        """
        penalty = disagreement_score * penalty_weight
        adjusted = confidence_v2 * (1 - penalty)

        return max(0.3, min(0.98, adjusted))


    @staticmethod
    def generate_disagreement_summary(
        segment_disagreements: Dict[int, DisagreementResult]
    ) -> Dict:
        """
        Generate summary of disagreement analysis

        Returns:
            {
                "total_segments": 25,
                "high_disagreement_count": 3,
                "review_required_count": 2,
                "avg_disagreement_score": 0.32,
                "high_disagreement_segments": [3, 7, 12],
                "consensus_distribution": {
                    "high": 18,
                    "medium": 5,
                    "low": 2
                }
            }
        """
        if not segment_disagreements:
            return {
                "total_segments": 0,
                "high_disagreement_count": 0,
                "review_required_count": 0,
                "avg_disagreement_score": 0.0,
                "high_disagreement_segments": [],
                "consensus_distribution": {
                    "high": 0,
                    "medium": 0,
                    "low": 0
                }
            }

        results = list(segment_disagreements.values())

        # Count consensus levels
        consensus_dist = {
            "high": sum(1 for r in results if r.consensus_level == "high"),
            "medium": sum(1 for r in results if r.consensus_level == "medium"),
            "low": sum(1 for r in results if r.consensus_level == "low")
        }

        # Find high disagreement segments
        high_disagreement = [
            r.segment_id for r in results
            if r.disagreement_score > 0.5
        ]

        # Review required segments
        review_required = [
            r.segment_id for r in results
            if r.requires_review
        ]

        # Average disagreement
        avg_disagreement = float(np.mean([r.disagreement_score for r in results]))

        return {
            "total_segments": len(results),
            "high_disagreement_count": len(high_disagreement),
            "review_required_count": len(review_required),
            "avg_disagreement_score": round(avg_disagreement, 3),
            "high_disagreement_segments": high_disagreement,
            "review_required_segments": review_required,
            "consensus_distribution": consensus_dist
        }


class DiversityMetrics:
    """
    Ensemble diversity metrics for multi-agent systems

    References:
        - EMNLP 2024: LLM-TOPLA
    """

    @staticmethod
    def compute_pairwise_agreement(
        agent_scores: List[List[float]]
    ) -> float:
        """
        Compute pairwise agreement between agents

        Args:
            agent_scores: List of score lists per agent
                [
                    [85, 72, 68, ...],  # Agent 1 scores
                    [82, 75, 65, ...],  # Agent 2 scores
                    ...
                ]

        Returns:
            Average pairwise correlation (0-1)
        """
        n_agents = len(agent_scores)

        if n_agents < 2:
            return 1.0

        correlations = []

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Pearson correlation
                corr = np.corrcoef(agent_scores[i], agent_scores[j])[0, 1]
                correlations.append(corr)

        avg_agreement = float(np.mean(correlations))

        return avg_agreement


    @staticmethod
    def compute_diversity_score(
        agent_scores: List[List[float]]
    ) -> float:
        """
        Diversity score = 1 - agreement

        Higher diversity → lower agreement
        """
        agreement = DiversityMetrics.compute_pairwise_agreement(agent_scores)
        diversity = 1 - agreement

        return diversity


    @staticmethod
    def compute_consensus_confidence(
        agent_confidences: List[float]
    ) -> Tuple[float, str]:
        """
        Consensus-based confidence

        High agreement → high confidence
        Low agreement → low confidence

        Returns:
            (consensus_confidence, consensus_level)
        """
        if not agent_confidences:
            return 0.5, "unknown"

        # Compute variance
        variance = float(np.var(agent_confidences))

        # Map variance to confidence
        # Low variance → high consensus → high confidence
        if variance < 0.01:
            consensus_level = "very_high"
            consensus_confidence = float(np.mean(agent_confidences))
        elif variance < 0.05:
            consensus_level = "high"
            consensus_confidence = float(np.mean(agent_confidences)) * 0.95
        elif variance < 0.1:
            consensus_level = "medium"
            consensus_confidence = float(np.mean(agent_confidences)) * 0.85
        else:
            consensus_level = "low"
            consensus_confidence = float(np.mean(agent_confidences)) * 0.70

        return consensus_confidence, consensus_level
