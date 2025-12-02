"""
Risk scoring engine for housing suitability assessment.
Implements multi-factor risk scoring instead of binary safe/unsafe classification.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    """Definition of a single risk factor."""
    name: str
    weight: float  # Importance weight (0-1)
    score_func: callable  # Function that takes value and returns risk score (0-1)
    description: str
    required: bool = False  # Whether this factor is required for assessment


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    overall_score: float  # 0 (safest) to 1 (most risky)
    risk_level: str  # "Low", "Moderate", "High", "Very High"
    confidence: float  # 0-1, based on data availability
    factor_scores: Dict[str, float]  # Individual factor scores
    missing_data: List[str]  # List of missing data layers
    reasoning: str  # Human-readable explanation


class RiskScoringEngine:
    """Multi-factor risk scoring engine for housing suitability."""
    
    def __init__(self):
        self.risk_factors = self._setup_risk_factors()
        self.risk_thresholds = {
            'low': 0.25,
            'moderate': 0.50,
            'high': 0.75
        }
    
    def _setup_risk_factors(self) -> Dict[str, RiskFactor]:
        """Define risk factors and their scoring functions."""
        
        def slope_risk_score(slope_deg: float) -> float:
            """Slope risk: exponential increase above 15°"""
            if slope_deg <= 10:
                return 0.0  # Very safe
            elif slope_deg <= 15:
                return 0.1  # Safe
            elif slope_deg <= 25:
                return 0.3 + (slope_deg - 15) * 0.02  # Moderate
            elif slope_deg <= 35:
                return 0.5 + (slope_deg - 25) * 0.03  # High
            else:
                return min(1.0, 0.8 + (slope_deg - 35) * 0.02)  # Very high
        
        def water_proximity_risk_score(distance_m: float) -> float:
            """Water proximity risk: higher risk closer to water bodies"""
            if distance_m >= 100:
                return 0.0  # Safe distance
            elif distance_m >= 50:
                return 0.2  # Moderate risk
            elif distance_m >= 20:
                return 0.5  # High risk
            else:
                return 0.8  # Very high risk (flood/erosion prone)
        
        def elevation_risk_score(elevation_m: float) -> float:
            """Elevation risk: very low elevations might have drainage issues"""
            if elevation_m < 100:
                return 0.3  # Potential drainage/flood issues
            elif elevation_m > 2000:
                return 0.2  # High altitude challenges
            else:
                return 0.0  # Normal elevation range
        
        return {
            'slope': RiskFactor(
                name="Terrain Slope",
                weight=0.5,  # Most important factor
                score_func=slope_risk_score,
                description="Risk from steep terrain and potential landslides",
                required=True
            ),
            'water_proximity': RiskFactor(
                name="Water Proximity", 
                weight=0.3,
                score_func=water_proximity_risk_score,
                description="Risk from proximity to rivers/water bodies (flooding, erosion)",
                required=False
            ),
            'elevation': RiskFactor(
                name="Elevation",
                weight=0.2,
                score_func=elevation_risk_score, 
                description="Risk from very low or very high elevations",
                required=False
            )
        }
    
    def assess_risk(self, analysis_data: Dict[str, Any]) -> RiskAssessment:
        """Perform comprehensive risk assessment."""
        
        factor_scores = {}
        weighted_scores = []
        total_weight = 0
        missing_data = []
        reasoning_parts = []
        
        # Evaluate each risk factor
        for factor_id, factor in self.risk_factors.items():
            value = self._get_factor_value(factor_id, analysis_data)
            
            if value is not None:
                score = factor.score_func(value)
                factor_scores[factor_id] = score
                weighted_scores.append(score * factor.weight)
                total_weight += factor.weight
                
                # Add to reasoning
                risk_level = self._score_to_level(score)
                reasoning_parts.append(f"{factor.name}: {risk_level} ({self._format_value(factor_id, value)})")
                
            else:
                missing_data.append(factor.name)
                if factor.required:
                    # Cannot assess without required data
                    return RiskAssessment(
                        overall_score=0.5,  # Unknown risk
                        risk_level="Unknown", 
                        confidence=0.0,
                        factor_scores={},
                        missing_data=missing_data,
                        reasoning=f"Cannot assess: missing required data ({factor.name})"
                    )
        
        # Calculate overall risk score
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = 0.5  # Default when no data
        
        # Determine confidence based on data availability
        available_factors = len(factor_scores)
        total_factors = len(self.risk_factors)
        confidence = available_factors / total_factors
        
        # Adjust confidence based on importance of missing factors
        if missing_data:
            important_missing = [name for factor_id, factor in self.risk_factors.items() 
                               if factor.name in missing_data and factor.weight > 0.3]
            if important_missing:
                confidence *= 0.7  # Reduce confidence for missing important factors
        
        # Generate risk level and reasoning
        risk_level = self._score_to_level(overall_score)
        
        if confidence < 0.5:
            risk_level += " (Low Confidence)"
        
        reasoning = "; ".join(reasoning_parts)
        if missing_data:
            reasoning += f". Missing: {', '.join(missing_data)}"
        
        return RiskAssessment(
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            factor_scores=factor_scores,
            missing_data=missing_data,
            reasoning=reasoning
        )
    
    def _get_factor_value(self, factor_id: str, analysis_data: Dict[str, Any]) -> Optional[float]:
        """Extract the appropriate value for a risk factor from analysis data."""
        mapping = {
            'slope': 'slope_deg',
            'water_proximity': 'distance_to_water_m', 
            'elevation': 'elevation_m'
        }
        
        key = mapping.get(factor_id)
        if key:
            return analysis_data.get(key)
        return None
    
    def _format_value(self, factor_id: str, value: float) -> str:
        """Format factor value for display."""
        if factor_id == 'slope':
            return f"{value:.1f}°"
        elif factor_id == 'water_proximity':
            return f"{value:.0f}m from water"
        elif factor_id == 'elevation':
            return f"{value:.0f}m elevation"
        else:
            return f"{value:.2f}"
    
    def _score_to_level(self, score: float) -> str:
        """Convert numerical score to risk level."""
        if score <= self.risk_thresholds['low']:
            return "Low Risk"
        elif score <= self.risk_thresholds['moderate']:
            return "Moderate Risk"
        elif score <= self.risk_thresholds['high']:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def get_recommendations(self, assessment: RiskAssessment) -> List[str]:
        """Generate actionable recommendations based on risk assessment."""
        recommendations = []
        
        if assessment.confidence < 0.5:
            recommendations.append("Obtain additional site data before making decisions")
        
        if assessment.overall_score <= 0.25:
            recommendations.append("Suitable for housing with standard construction practices")
        
        elif assessment.overall_score <= 0.5:
            recommendations.append("Suitable with engineering assessment and mitigation measures")
            
            # Specific recommendations based on factor scores
            for factor_id, score in assessment.factor_scores.items():
                if score > 0.3:
                    if factor_id == 'slope':
                        recommendations.append("• Consider terracing or retaining walls for slope stability")
                    elif factor_id == 'water_proximity':
                        recommendations.append("• Implement drainage and flood protection measures")
                    elif factor_id == 'elevation':
                        recommendations.append("• Address elevation-related challenges (drainage/access)")
        
        elif assessment.overall_score <= 0.75:
            recommendations.append("High risk - extensive mitigation required")
            recommendations.append("• Professional geotechnical assessment mandatory")
            recommendations.append("• Consider alternative sites if possible")
        
        else:
            recommendations.append("Very high risk - not recommended for housing")
            recommendations.append("• Find alternative location")
            recommendations.append("• If no alternatives, extensive engineering required")
        
        return recommendations


# Factory function for easy instantiation
def create_risk_engine() -> RiskScoringEngine:
    """Create a configured risk scoring engine."""
    return RiskScoringEngine()