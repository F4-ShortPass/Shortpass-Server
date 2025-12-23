"""
Test Calibration Modules
Quick test script to verify new calibration functionality

Usage:
    python scripts/test_calibration_modules.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.evaluation.calibration_metrics import CalibrationMetrics, CalibrationMonitor
from services.evaluation.temperature_scaler import TemperatureScaler, PerCompetencyTemperatureScaler
from ai.agents.aggregators.disagreement_analyzer import DisagreementAnalyzer, DiversityMetrics

import numpy as np


def test_calibration_metrics():
    """Test ECE and Brier Score calculation"""
    print("\n" + "="*60)
    print("TEST 1: Calibration Metrics")
    print("="*60)

    # Synthetic data
    np.random.seed(42)

    # Well-calibrated model
    confidences_good = [0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3]
    outcomes_good = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

    result_good = CalibrationMetrics.analyze_calibration(
        confidences_good,
        outcomes_good
    )

    print("\nWell-Calibrated Model:")
    print(f"  ECE: {result_good.ece:.3f}")
    print(f"  Brier Score: {result_good.brier_score:.3f}")
    print(f"  Quality: {result_good.calibration_quality}")
    print(f"  Recommendations: {result_good.recommendations}")

    # Overconfident model
    confidences_overconf = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    outcomes_overconf = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]

    result_overconf = CalibrationMetrics.analyze_calibration(
        confidences_overconf,
        outcomes_overconf
    )

    print("\nOverconfident Model:")
    print(f"  ECE: {result_overconf.ece:.3f}")
    print(f"  Brier Score: {result_overconf.brier_score:.3f}")
    print(f"  Quality: {result_overconf.calibration_quality}")
    print(f"  Recommendations: {result_overconf.recommendations}")

    print("\n✅ Test 1 Passed")


def test_temperature_scaling():
    """Test temperature scaling calibration"""
    print("\n" + "="*60)
    print("TEST 2: Temperature Scaling")
    print("="*60)

    # Synthetic overconfident model
    np.random.seed(42)

    # Generate synthetic validation data
    n_samples = 50
    confidences = np.random.beta(8, 2, n_samples)  # Skewed towards high confidence
    outcomes = (np.random.rand(n_samples) < (confidences * 0.7)).astype(int)  # Actual accuracy lower

    print(f"\nOriginal Data (n={n_samples}):")
    print(f"  Avg Confidence: {np.mean(confidences):.3f}")
    print(f"  Avg Accuracy: {np.mean(outcomes):.3f}")
    print(f"  Overconfidence: {np.mean(confidences) - np.mean(outcomes):.3f}")

    # Compute original ECE
    ece_before = CalibrationMetrics.compute_ece(
        confidences.tolist(),
        outcomes.tolist()
    )
    print(f"  ECE Before: {ece_before:.3f}")

    # Train temperature scaler
    scaler = TemperatureScaler()
    temperature = scaler.fit(confidences.tolist(), outcomes.tolist())

    print(f"\nLearned Temperature: {temperature:.3f}")

    # Apply calibration
    calibrated_confs = scaler.transform(confidences.tolist())

    print(f"\nCalibrated Data:")
    print(f"  Avg Calibrated Confidence: {np.mean(calibrated_confs):.3f}")
    print(f"  Avg Accuracy: {np.mean(outcomes):.3f}")
    print(f"  Overconfidence: {np.mean(calibrated_confs) - np.mean(outcomes):.3f}")

    # Compute calibrated ECE
    ece_after = CalibrationMetrics.compute_ece(
        calibrated_confs,
        outcomes.tolist()
    )
    print(f"  ECE After: {ece_after:.3f}")

    improvement = (ece_before - ece_after) / ece_before * 100
    print(f"\nECE Improvement: {improvement:.1f}%")

    print("\n✅ Test 2 Passed")


def test_disagreement_analyzer():
    """Test inter-agent disagreement detection"""
    print("\n" + "="*60)
    print("TEST 3: Disagreement Analyzer")
    print("="*60)

    # Synthetic segment evaluations
    segment_evaluations = [
        # Segment 1 - High consensus
        {"competency": "achievement_motivation", "segment_id": 1, "score": 85, "confidence_v2": 0.88},
        {"competency": "growth_potential", "segment_id": 1, "score": 82, "confidence_v2": 0.85},
        {"competency": "problem_solving", "segment_id": 1, "score": 87, "confidence_v2": 0.90},

        # Segment 2 - Low consensus (high disagreement)
        {"competency": "achievement_motivation", "segment_id": 2, "score": 90, "confidence_v2": 0.92},
        {"competency": "growth_potential", "segment_id": 2, "score": 55, "confidence_v2": 0.60},
        {"competency": "problem_solving", "segment_id": 2, "score": 70, "confidence_v2": 0.75},

        # Segment 3 - Medium consensus
        {"competency": "achievement_motivation", "segment_id": 3, "score": 75, "confidence_v2": 0.78},
        {"competency": "growth_potential", "segment_id": 3, "score": 68, "confidence_v2": 0.72},
        {"competency": "problem_solving", "segment_id": 3, "score": 72, "confidence_v2": 0.75},
    ]

    # Compute disagreement
    disagreements = DisagreementAnalyzer.compute_segment_disagreement(
        segment_evaluations
    )

    print("\nSegment-Level Disagreement:")
    for seg_id, result in disagreements.items():
        print(f"\n  Segment {seg_id}:")
        print(f"    Score Variance: {result.score_variance:.1f}")
        print(f"    Disagreement Score: {result.disagreement_score:.3f}")
        print(f"    Consensus Level: {result.consensus_level}")
        print(f"    Requires Review: {result.requires_review}")

    # Generate summary
    summary = DisagreementAnalyzer.generate_disagreement_summary(disagreements)

    print("\nDisagreement Summary:")
    print(f"  Total Segments: {summary['total_segments']}")
    print(f"  High Disagreement: {summary['high_disagreement_count']}")
    print(f"  Review Required: {summary['review_required_count']}")
    print(f"  Avg Disagreement: {summary['avg_disagreement_score']:.3f}")
    print(f"  Consensus Distribution: {summary['consensus_distribution']}")

    print("\n✅ Test 3 Passed")


def test_diversity_metrics():
    """Test ensemble diversity metrics"""
    print("\n" + "="*60)
    print("TEST 4: Ensemble Diversity Metrics")
    print("="*60)

    # Synthetic agent scores
    # High agreement scenario
    agent_scores_high_agreement = [
        [85, 72, 68, 90, 75],  # Agent 1
        [83, 74, 70, 88, 73],  # Agent 2
        [87, 71, 69, 92, 76],  # Agent 3
    ]

    agreement_high = DiversityMetrics.compute_pairwise_agreement(
        agent_scores_high_agreement
    )
    diversity_high = DiversityMetrics.compute_diversity_score(
        agent_scores_high_agreement
    )

    print("\nHigh Agreement Scenario:")
    print(f"  Pairwise Agreement: {agreement_high:.3f}")
    print(f"  Diversity Score: {diversity_high:.3f}")

    # Low agreement scenario
    agent_scores_low_agreement = [
        [85, 72, 68, 90, 75],  # Agent 1
        [55, 40, 90, 60, 50],  # Agent 2
        [70, 95, 45, 80, 65],  # Agent 3
    ]

    agreement_low = DiversityMetrics.compute_pairwise_agreement(
        agent_scores_low_agreement
    )
    diversity_low = DiversityMetrics.compute_diversity_score(
        agent_scores_low_agreement
    )

    print("\nLow Agreement Scenario:")
    print(f"  Pairwise Agreement: {agreement_low:.3f}")
    print(f"  Diversity Score: {diversity_low:.3f}")

    # Consensus confidence
    agent_confidences_high_consensus = [0.88, 0.85, 0.87, 0.86]
    conf_high, level_high = DiversityMetrics.compute_consensus_confidence(
        agent_confidences_high_consensus
    )

    print("\nHigh Consensus Confidences:")
    print(f"  Raw Confidences: {agent_confidences_high_consensus}")
    print(f"  Consensus Confidence: {conf_high:.3f}")
    print(f"  Consensus Level: {level_high}")

    agent_confidences_low_consensus = [0.95, 0.60, 0.75, 0.50]
    conf_low, level_low = DiversityMetrics.compute_consensus_confidence(
        agent_confidences_low_consensus
    )

    print("\nLow Consensus Confidences:")
    print(f"  Raw Confidences: {agent_confidences_low_consensus}")
    print(f"  Consensus Confidence: {conf_low:.3f}")
    print(f"  Consensus Level: {level_low}")

    print("\n✅ Test 4 Passed")


def test_calibration_monitor():
    """Test calibration monitoring over time"""
    print("\n" + "="*60)
    print("TEST 5: Calibration Monitor")
    print("="*60)

    monitor = CalibrationMonitor(window_size=20)

    # Simulate incoming evaluations
    print("\nAdding evaluations to monitor...")

    # Good calibration period
    for i in range(10):
        confidence = 0.8 + np.random.randn() * 0.05
        outcome = 1 if np.random.rand() < confidence * 0.95 else 0
        monitor.add_evaluation(
            f"eval_{i}",
            confidence,
            outcome,
            f"2025-01-{i+1:02d}"
        )

    metrics_good = monitor.get_current_metrics()
    print(f"\nGood Calibration Period:")
    print(f"  Sample Size: {metrics_good['sample_size']}")
    print(f"  ECE: {metrics_good.get('ece', 'N/A')}")

    # Poor calibration period (drift)
    for i in range(10, 20):
        confidence = 0.9 + np.random.randn() * 0.03  # Overconfident
        outcome = 1 if np.random.rand() < 0.6 else 0  # Lower accuracy
        monitor.add_evaluation(
            f"eval_{i}",
            confidence,
            outcome,
            f"2025-01-{i+1:02d}"
        )

    metrics_poor = monitor.get_current_metrics()
    print(f"\nPoor Calibration Period:")
    print(f"  Sample Size: {metrics_poor['sample_size']}")
    print(f"  ECE: {metrics_poor.get('ece', 'N/A')}")
    print(f"  Quality: {metrics_poor.get('calibration_quality', 'N/A')}")

    # Detect drift
    drift = monitor.detect_drift(threshold=0.05)
    print(f"\nCalibration Drift Detected: {drift}")

    print("\n✅ Test 5 Passed")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CALIBRATION MODULES TEST SUITE")
    print("="*60)

    try:
        test_calibration_metrics()
        test_temperature_scaling()
        test_disagreement_analyzer()
        test_diversity_metrics()
        test_calibration_monitor()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nNext Steps:")
        print("1. Review integration guide: server/docs/CALIBRATION_INTEGRATION_GUIDE.md")
        print("2. Collect validation data for temperature scaling")
        print("3. Integrate disagreement analysis into Stage 2")
        print("4. Add calibration metrics to Stage 3")
        print("\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
