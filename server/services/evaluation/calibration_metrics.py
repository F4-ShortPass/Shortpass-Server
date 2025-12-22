"""
Calibration Metrics for LLM-as-a-Judge Evaluation
Expected Calibration Error (ECE), Brier Score, Reliability Diagrams

References:
    - EMNLP 2024: Adaptive Temperature Scaling
    - ICML 2024: Thermometer Framework
    - arXiv 2024: Conformal Prediction for LLM-as-a-Judge
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Calibration metrics result"""
    ece: float  # Expected Calibration Error
    brier_score: float  # Brier Score
    reliability_diagram: Dict  # Data for visualization
    calibration_quality: str  # "excellent"/"good"/"fair"/"poor"
    recommendations: List[str]  # Actionable insights


class CalibrationMetrics:
    """
    Calibration metrics for LLM-as-a-judge evaluation

    Measures alignment between predicted confidence and actual accuracy
    """

    # Quality thresholds
    ECE_EXCELLENT = 0.05
    ECE_GOOD = 0.10
    ECE_FAIR = 0.15

    BRIER_EXCELLENT = 0.10
    BRIER_GOOD = 0.20
    BRIER_FAIR = 0.30


    @staticmethod
    def compute_ece(
        confidences: List[float],
        accuracies: List[float],
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE)

        Measures the weighted average of absolute differences between
        predicted confidence and empirical accuracy across bins.

        Args:
            confidences: Predicted confidence scores (0-1)
            accuracies: Actual binary outcomes (0 or 1)
            n_bins: Number of bins for calibration curve

        Returns:
            ECE score (lower is better, range [0, 1])

        Example:
            >>> confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
            >>> accuracies = [1, 1, 0, 1, 0]
            >>> ece = CalibrationMetrics.compute_ece(confidences, accuracies)
            >>> print(f"ECE: {ece:.3f}")
        """
        if len(confidences) == 0:
            return 0.0

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(confidences[in_bin])
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)

                ece += bin_weight * np.abs(bin_confidence - bin_accuracy)

        return float(ece)


    @staticmethod
    def compute_brier_score(
        confidences: List[float],
        outcomes: List[float]
    ) -> float:
        """
        Brier Score

        Mean squared error between predicted probabilities and actual outcomes.
        Penalizes overconfident incorrect predictions.

        Args:
            confidences: Predicted probabilities (0-1)
            outcomes: Actual binary outcomes (0 or 1)

        Returns:
            Brier score (lower is better, range [0, 1])

        Example:
            >>> confidences = [0.9, 0.8, 0.7]
            >>> outcomes = [1, 1, 0]
            >>> brier = CalibrationMetrics.compute_brier_score(confidences, outcomes)
            >>> print(f"Brier: {brier:.3f}")
        """
        if len(confidences) == 0:
            return 0.0

        confidences = np.array(confidences)
        outcomes = np.array(outcomes)

        return float(np.mean((confidences - outcomes) ** 2))


    @staticmethod
    def compute_reliability_diagram(
        confidences: List[float],
        accuracies: List[float],
        n_bins: int = 10
    ) -> Dict:
        """
        Reliability diagram data for visualization

        Returns data for plotting predicted confidence vs. empirical accuracy.
        Perfectly calibrated model would have all points on y=x diagonal.

        Args:
            confidences: Predicted confidence scores
            accuracies: Actual binary outcomes
            n_bins: Number of bins

        Returns:
            {
                "bin_centers": [0.05, 0.15, ..., 0.95],
                "bin_accuracies": [0.03, 0.12, ..., 0.94],
                "bin_counts": [5, 12, ..., 8],
                "perfect_calibration": [0.05, 0.15, ..., 0.95]  # y=x line
            }
        """
        if len(confidences) == 0:
            return {
                "bin_centers": [],
                "bin_accuracies": [],
                "bin_counts": [],
                "perfect_calibration": []
            }

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies_list = []
        bin_counts = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_center = (bin_lower + bin_upper) / 2
                bin_centers.append(bin_center)
                bin_accuracies_list.append(float(np.mean(accuracies[in_bin])))
                bin_counts.append(int(np.sum(in_bin)))

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies_list,
            "bin_counts": bin_counts,
            "perfect_calibration": bin_centers  # Perfect calibration is y=x
        }


    @staticmethod
    def analyze_calibration(
        confidences: List[float],
        outcomes: List[float],
        n_bins: int = 10
    ) -> CalibrationResult:
        """
        Comprehensive calibration analysis

        Args:
            confidences: Predicted confidence scores
            outcomes: Actual binary outcomes
            n_bins: Number of bins for ECE

        Returns:
            CalibrationResult with all metrics and recommendations
        """
        ece = CalibrationMetrics.compute_ece(confidences, outcomes, n_bins)
        brier = CalibrationMetrics.compute_brier_score(confidences, outcomes)
        reliability = CalibrationMetrics.compute_reliability_diagram(
            confidences, outcomes, n_bins
        )

        # Determine calibration quality
        if ece <= CalibrationMetrics.ECE_EXCELLENT and brier <= CalibrationMetrics.BRIER_EXCELLENT:
            quality = "excellent"
        elif ece <= CalibrationMetrics.ECE_GOOD and brier <= CalibrationMetrics.BRIER_GOOD:
            quality = "good"
        elif ece <= CalibrationMetrics.ECE_FAIR and brier <= CalibrationMetrics.BRIER_FAIR:
            quality = "fair"
        else:
            quality = "poor"

        # Generate recommendations
        recommendations = []

        if ece > CalibrationMetrics.ECE_GOOD:
            recommendations.append(
                f"ECE ({ece:.3f}) is high. Consider temperature scaling or isotonic regression."
            )

        if brier > CalibrationMetrics.BRIER_GOOD:
            recommendations.append(
                f"Brier Score ({brier:.3f}) is high. Review prediction accuracy and confidence alignment."
            )

        if len(confidences) < 30:
            recommendations.append(
                f"Sample size ({len(confidences)}) is low. Collect more validation data for reliable calibration."
            )

        # Check for overconfidence
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0
        avg_accuracy = np.mean(outcomes) if len(outcomes) > 0 else 0

        if avg_confidence - avg_accuracy > 0.15:
            recommendations.append(
                f"Model is overconfident (conf={avg_confidence:.2f}, acc={avg_accuracy:.2f}). Apply calibration."
            )
        elif avg_accuracy - avg_confidence > 0.15:
            recommendations.append(
                f"Model is underconfident (conf={avg_confidence:.2f}, acc={avg_accuracy:.2f}). Review confidence calculation."
            )

        return CalibrationResult(
            ece=ece,
            brier_score=brier,
            reliability_diagram=reliability,
            calibration_quality=quality,
            recommendations=recommendations
        )


    @staticmethod
    def compute_per_competency_calibration(
        evaluations: List[Dict],
        competency_names: List[str]
    ) -> Dict[str, CalibrationResult]:
        """
        Compute calibration metrics per competency

        Args:
            evaluations: List of evaluation results
                [
                    {
                        "competency_name": "achievement_motivation",
                        "confidence_v2": 0.85,
                        "ground_truth_pass": true
                    },
                    ...
                ]
            competency_names: List of competency names to analyze

        Returns:
            {competency_name: CalibrationResult}
        """
        results = {}

        for comp_name in competency_names:
            # Filter evaluations for this competency
            comp_evals = [
                e for e in evaluations
                if e.get("competency_name") == comp_name
            ]

            if len(comp_evals) < 5:
                # Not enough data for reliable calibration
                results[comp_name] = CalibrationResult(
                    ece=0.0,
                    brier_score=0.0,
                    reliability_diagram={
                        "bin_centers": [],
                        "bin_accuracies": [],
                        "bin_counts": [],
                        "perfect_calibration": []
                    },
                    calibration_quality="insufficient_data",
                    recommendations=[f"Need at least 5 samples for {comp_name}, got {len(comp_evals)}"]
                )
                continue

            # Extract confidences and outcomes
            confidences = [e.get("confidence_v2", 0.5) for e in comp_evals]
            outcomes = [float(e.get("ground_truth_pass", 0)) for e in comp_evals]

            results[comp_name] = CalibrationMetrics.analyze_calibration(
                confidences, outcomes
            )

        return results


class CalibrationMonitor:
    """
    Monitor calibration metrics over time

    Tracks ECE and Brier Score trends to detect calibration drift
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history = []

    def add_evaluation(
        self,
        evaluation_id: str,
        confidence: float,
        outcome: float,
        timestamp: str
    ):
        """Add evaluation to monitoring history"""
        self.history.append({
            "evaluation_id": evaluation_id,
            "confidence": confidence,
            "outcome": outcome,
            "timestamp": timestamp
        })

        # Keep only recent window
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def get_current_metrics(self) -> Dict:
        """Get calibration metrics for current window"""
        if len(self.history) < 10:
            return {
                "status": "insufficient_data",
                "sample_size": len(self.history)
            }

        confidences = [h["confidence"] for h in self.history]
        outcomes = [h["outcome"] for h in self.history]

        result = CalibrationMetrics.analyze_calibration(confidences, outcomes)

        return {
            "status": "ok",
            "sample_size": len(self.history),
            "ece": result.ece,
            "brier_score": result.brier_score,
            "calibration_quality": result.calibration_quality,
            "recommendations": result.recommendations
        }

    def detect_drift(self, threshold: float = 0.05) -> bool:
        """
        Detect calibration drift

        Returns True if recent ECE is significantly worse than historical
        """
        if len(self.history) < 30:
            return False

        # Split into two halves
        mid = len(self.history) // 2

        first_half = self.history[:mid]
        second_half = self.history[mid:]

        ece_first = CalibrationMetrics.compute_ece(
            [h["confidence"] for h in first_half],
            [h["outcome"] for h in first_half]
        )

        ece_second = CalibrationMetrics.compute_ece(
            [h["confidence"] for h in second_half],
            [h["outcome"] for h in second_half]
        )

        # Drift detected if ECE increased by threshold
        return (ece_second - ece_first) > threshold
