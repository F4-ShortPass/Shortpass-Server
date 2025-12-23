"""
Temperature Scaling for Confidence Calibration
Post-hoc calibration method for LLM-as-a-judge systems

References:
    - Guo et al. 2017: On Calibration of Modern Neural Networks
    - EMNLP 2024: Adaptive Temperature Scaling
    - ICML 2024: Thermometer Framework
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize
import json
import os


class TemperatureScaler:
    """
    Post-hoc temperature scaling for confidence calibration

    Temperature scaling applies a single scalar parameter to calibrate
    predicted confidences without changing the rank order of predictions.

    Usage:
        # Training
        scaler = TemperatureScaler()
        scaler.fit(validation_confidences, validation_outcomes)

        # Inference
        calibrated_conf = scaler.transform(test_confidences)
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Initial temperature (1.0 = no scaling)
        """
        self.temperature = temperature
        self.is_fitted = False


    def fit(
        self,
        confidences: List[float],
        outcomes: List[int],
        method: str = "nll"
    ) -> float:
        """
        Learn optimal temperature on validation set

        Args:
            confidences: Raw confidence scores (uncalibrated, 0-1)
            outcomes: Ground truth binary outcomes (0 or 1)
            method: Optimization objective ("nll" or "ece")

        Returns:
            Optimal temperature value
        """
        confidences = np.array(confidences)
        outcomes = np.array(outcomes)

        if len(confidences) < 10:
            print("Warning: Too few samples for temperature scaling. Using default T=1.0")
            self.temperature = 1.0
            self.is_fitted = True
            return self.temperature

        # Convert to logits (inverse sigmoid)
        # Handle edge cases for numerical stability
        confidences_clipped = np.clip(confidences, 1e-7, 1 - 1e-7)
        logits = np.log(confidences_clipped / (1 - confidences_clipped))

        def objective(temp):
            """Optimization objective"""
            if method == "nll":
                return self._nll_loss(logits, outcomes, temp)
            elif method == "ece":
                return self._ece_loss(logits, outcomes, temp)
            else:
                raise ValueError(f"Unknown method: {method}")

        # Optimize temperature
        result = minimize(
            objective,
            x0=1.0,
            bounds=[(0.1, 10.0)],
            method='L-BFGS-B'
        )

        self.temperature = float(result.x[0])
        self.is_fitted = True

        print(f"Temperature Scaling: Optimal T = {self.temperature:.3f}")

        return self.temperature


    def _nll_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temp: float
    ) -> float:
        """
        Negative log-likelihood loss

        Minimizing NLL is equivalent to maximizing likelihood
        of correct predictions.
        """
        # Scale logits by temperature
        scaled_logits = logits / temp

        # Convert to probabilities (sigmoid)
        probs = 1 / (1 + np.exp(-scaled_logits))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)

        # Binary cross-entropy
        loss = -np.mean(
            labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
        )

        return loss


    def _ece_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        temp: float,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error loss

        Directly optimize for calibration quality.
        """
        # Scale logits
        scaled_logits = logits / temp
        probs = 1 / (1 + np.exp(-scaled_logits))

        # Compute ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (probs > bin_lower) & (probs <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(probs[in_bin])
                bin_accuracy = np.mean(labels[in_bin])
                bin_weight = np.sum(in_bin) / len(probs)

                ece += bin_weight * np.abs(bin_confidence - bin_accuracy)

        return ece


    def transform(
        self,
        confidences: List[float]
    ) -> List[float]:
        """
        Apply learned temperature scaling

        Args:
            confidences: Uncalibrated confidence scores (0-1)

        Returns:
            Calibrated confidence scores (0-1)
        """
        if not self.is_fitted:
            print("Warning: TemperatureScaler not fitted. Returning original confidences.")
            return confidences

        confidences = np.array(confidences)

        # Convert to logits
        confidences_clipped = np.clip(confidences, 1e-7, 1 - 1e-7)
        logits = np.log(confidences_clipped / (1 - confidences_clipped))

        # Scale by temperature
        scaled_logits = logits / self.temperature

        # Convert back to probabilities
        calibrated = 1 / (1 + np.exp(-scaled_logits))

        return calibrated.tolist()


    def save(self, filepath: str):
        """Save temperature to file"""
        config = {
            "temperature": self.temperature,
            "is_fitted": self.is_fitted
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


    @classmethod
    def load(cls, filepath: str) -> 'TemperatureScaler':
        """Load temperature from file"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        scaler = cls(temperature=config["temperature"])
        scaler.is_fitted = config["is_fitted"]

        return scaler


class PerCompetencyTemperatureScaler:
    """
    Learn separate temperature for each competency

    Different competencies may have different calibration characteristics.
    E.g., behavioral competencies might be less confident than technical ones.
    """

    def __init__(self):
        self.scalers: Dict[str, TemperatureScaler] = {}
        self.default_scaler = TemperatureScaler()


    def fit(
        self,
        competencies: List[str],
        confidences_by_comp: Dict[str, List[float]],
        outcomes_by_comp: Dict[str, List[int]]
    ) -> Dict[str, float]:
        """
        Learn separate temperature for each competency

        Args:
            competencies: List of competency names
            confidences_by_comp: {competency: [confidence scores]}
            outcomes_by_comp: {competency: [binary outcomes]}

        Returns:
            {competency_name: temperature}
        """
        temperatures = {}

        for comp in competencies:
            if comp not in confidences_by_comp or len(confidences_by_comp[comp]) < 10:
                print(f"Skipping {comp}: insufficient data")
                continue

            scaler = TemperatureScaler()
            temp = scaler.fit(
                confidences_by_comp[comp],
                outcomes_by_comp[comp]
            )

            self.scalers[comp] = scaler
            temperatures[comp] = temp

        # Fit default scaler on all data
        all_confidences = []
        all_outcomes = []

        for comp in competencies:
            if comp in confidences_by_comp:
                all_confidences.extend(confidences_by_comp[comp])
                all_outcomes.extend(outcomes_by_comp[comp])

        if len(all_confidences) >= 10:
            self.default_scaler.fit(all_confidences, all_outcomes)

        return temperatures


    def transform(
        self,
        competency: str,
        confidences: List[float]
    ) -> List[float]:
        """
        Apply competency-specific temperature scaling

        Args:
            competency: Competency name
            confidences: Uncalibrated confidences

        Returns:
            Calibrated confidences
        """
        scaler = self.scalers.get(competency, self.default_scaler)
        return scaler.transform(confidences)


    def save(self, directory: str):
        """Save all scalers to directory"""
        os.makedirs(directory, exist_ok=True)

        # Save default scaler
        self.default_scaler.save(os.path.join(directory, "default.json"))

        # Save per-competency scalers
        for comp, scaler in self.scalers.items():
            filename = f"{comp}.json"
            scaler.save(os.path.join(directory, filename))


    @classmethod
    def load(cls, directory: str) -> 'PerCompetencyTemperatureScaler':
        """Load all scalers from directory"""
        instance = cls()

        # Load default
        default_path = os.path.join(directory, "default.json")
        if os.path.exists(default_path):
            instance.default_scaler = TemperatureScaler.load(default_path)

        # Load per-competency
        for filename in os.listdir(directory):
            if filename == "default.json":
                continue

            if filename.endswith(".json"):
                comp_name = filename.replace(".json", "")
                filepath = os.path.join(directory, filename)
                instance.scalers[comp_name] = TemperatureScaler.load(filepath)

        return instance


class AdaptiveTemperatureScaler:
    """
    Adaptive temperature scaling that adjusts based on input features

    References:
        - EMNLP 2024: Adaptive Temperature Scaling

    Instead of a single temperature, predicts temperature based on:
        - Competency type
        - Resume verification strength
        - Segment count
        - Interview length
    """

    def __init__(self):
        self.feature_weights = {}
        self.base_temperature = 1.0


    def predict_temperature(
        self,
        competency_type: str,  # "common" or "job"
        verification_strength: str,  # "high"/"medium"/"low"/"none"
        segment_count: int,
        interview_length: int
    ) -> float:
        """
        Predict adaptive temperature based on input features

        Heuristic approach:
            - High verification → lower temperature (more confident)
            - Many segments → lower temperature (more evidence)
            - Behavioral competencies → higher temperature (less certain)
        """
        temp = self.base_temperature

        # Verification strength adjustment
        verification_adj = {
            "high": -0.2,
            "medium": -0.1,
            "low": 0.0,
            "none": 0.1
        }.get(verification_strength, 0.0)

        # Segment count adjustment (more evidence → lower temp)
        segment_adj = -0.05 * min(segment_count / 5, 1.0)

        # Competency type adjustment
        type_adj = 0.1 if competency_type == "common" else 0.0

        temp += verification_adj + segment_adj + type_adj

        # Clamp to reasonable range
        temp = max(0.5, min(temp, 2.0))

        return temp
