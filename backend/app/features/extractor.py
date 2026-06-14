"""Visual feature extraction pipeline"""
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from app.models.schemas import FeatureAnalysis

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts visual features from images for geolocation inference.
    
    This module implements the 50+ visual cues pipeline as described in the system design:
    - Natural environment (vegetation, soil, terrain)
    - Weather & astronomy (sun, sky, clouds, stars)
    - Infrastructure (roads, poles, signage)
    - Architecture (roofs, walls, facades)
    - Cultural indicators (clothing, symbols, text)
    """
    
    def __init__(self):
        """Initialize feature extractors"""
        self.landmark_detector_ready = False
        self.vegetation_classifier_ready = False
        self.text_detector_ready = False
        logger.info("FeatureExtractor initialized")
    
    def extract_all_features(self, image_array: np.ndarray) -> FeatureAnalysis:
        """
        Extract all available visual features from an image.
        
        Args:
            image_array: Input image as numpy array (H, W, 3)
            
        Returns:
            FeatureAnalysis object with detected features
        """
        try:
            from app.config import settings

            landmarks = self._detect_landmarks(image_array)
            if getattr(settings, "feature_analysis_clip_landmarks", True):
                clip_lm = self._detect_landmarks_clip(image_array)
                if clip_lm:
                    landmarks = (landmarks or []) + clip_lm

            architecture = self._classify_architecture(image_array)
            if getattr(settings, "feature_analysis_clip_architecture", True):
                clip_arch = self._classify_architecture_clip(image_array)
                if clip_arch:
                    architecture = clip_arch

            detected_text = self._detect_text(image_array)
            if getattr(settings, "feature_analysis_ocr_enabled", True):
                ocr_text = self._detect_text_ocr(image_array)
                if ocr_text:
                    detected_text = (detected_text or []) + ocr_text

            features = {
                "landmarks": landmarks,
                "vegetation_types": self._analyze_vegetation(image_array),
                "architecture_style": architecture,
                "detected_text": detected_text,
                "weather_condition": self._classify_weather(image_array),
                "time_of_day": self._estimate_time_of_day(image_array),
                "infrastructure_type": self._classify_infrastructure(image_array),
            }
            
            # Run specialist detectors (poles, road lines, shadows)
            try:
                from app.features.specialist_detectors import (
                    detect_pole_proxies,
                    detect_road_line_proxies,
                    detect_shadow_features,
                )
                poles = detect_pole_proxies(image_array)
                lines = detect_road_line_proxies(image_array)
                shadows = detect_shadow_features(image_array)
                
                if poles:
                    features["detected_poles"] = poles
                if lines:
                    features["detected_road_lines"] = lines
                if shadows.get("confidence", 0) > 0.2:
                    features["shadow_analysis"] = shadows
                    
            except Exception as e:
                logger.debug("Specialist detectors skipped: %s", e)
            
            logger.info("Extracted %d feature groups", sum(1 for v in features.values() if v))
            return FeatureAnalysis(**features)
            
        except Exception as e:
            logger.error("Error extracting features: %s", e)
            return FeatureAnalysis()
    
    def _detect_landmarks(self, image_array: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """Reserved for a dedicated landmark detector; CLIP hints use _detect_landmarks_clip."""
        return None

    def _detect_landmarks_clip(self, image_array: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        from app.features.clip_cue_hints import clip_landmark_hints

        hints = clip_landmark_hints(image_array)
        return hints if hints else None
    
    def _water_fraction_central(self, image_array: np.ndarray) -> float:
        """Share of blue water-like pixels in the central frame (lakes read as low green)."""
        if image_array is None or image_array.size == 0:
            return 0.0
        try:
            from app.features.water_pixels import water_fraction_central

            return water_fraction_central(image_array)
        except Exception:
            return 0.0

    def _analyze_vegetation(self, image_array: np.ndarray) -> Optional[List[str]]:
        """
        Lightweight vegetation hints from green ratio — not species ID.
        Skips arid labels when open water dominates (common lake false positive).
        """
        water_frac = self._water_fraction_central(image_array)
        green_ratio = self._calculate_green_ratio(image_array)
        vegetation: List[str] = []

        if water_frac >= 0.08:
            vegetation.append("open_water_or_lake")
            if green_ratio > 0.12:
                vegetation.append("temperate_lake_shore")
            return vegetation

        if green_ratio > 0.4:
            vegetation.append("dense_vegetation")
        if green_ratio > 0.6:
            vegetation.append("tropical_climate")
        elif green_ratio < 0.2:
            vegetation.append("low_green_cover")

        return vegetation if vegetation else None
    
    def _classify_architecture(self, image_array: np.ndarray) -> Optional[str]:
        """Heuristic placeholder; CLIP style via _classify_architecture_clip."""
        return None

    def _classify_architecture_clip(self, image_array: np.ndarray) -> Optional[str]:
        from app.features.clip_cue_hints import clip_architecture_hint

        return clip_architecture_hint(image_array)

    def _detect_text(self, image_array: np.ndarray) -> Optional[List[str]]:
        """Reserved for learned OCR; Tesseract via _detect_text_ocr."""
        return None

    def _detect_text_ocr(self, image_array: np.ndarray) -> Optional[List[str]]:
        from app.features.ocr_text import detect_text_in_image

        return detect_text_in_image(image_array)
    
    def _classify_weather(self, image_array: np.ndarray) -> Optional[str]:
        """
        Classify weather conditions in the image.
        
        Examples: sunny, cloudy, rainy, foggy, snowy, etc.
        """
        # Mock implementation using sky color analysis
        sky_brightness = self._estimate_sky_brightness(image_array)
        
        if sky_brightness > 0.7:
            return "sunny"
        elif sky_brightness > 0.5:
            return "partly_cloudy"
        elif sky_brightness > 0.3:
            return "cloudy"
        else:
            return "overcast"
    
    def _estimate_time_of_day(self, image_array: np.ndarray) -> Optional[str]:
        """
        Estimate time of day from lighting conditions.
        
        Uses: Shadow direction, light color, sky hue.
        Returns: sunrise, morning, afternoon, sunset, night
        """
        # Mock implementation
        brightness = self._estimate_overall_brightness(image_array)
        
        if brightness > 0.8:
            return "midday"
        elif brightness > 0.5:
            return "afternoon"
        elif brightness > 0.3:
            return "evening"
        else:
            return "night"
    
    def _classify_infrastructure(self, image_array: np.ndarray) -> Optional[str]:
        """
        Classify infrastructure type visible in image.
        
        Examples: urban, suburban, rural, highway, street_level
        """
        # Mock implementation
        if self._has_high_contrast(image_array):
            return "urban"
        else:
            return "rural"
    
    # Helper methods for feature extraction
    
    def _calculate_green_ratio(self, image_array: np.ndarray) -> float:
        """Calculate ratio of green pixels (vegetation indicator)"""
        if len(image_array) == 0:
            return 0.0
        
        # Extract channels
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        
        # Simple green detection: g > r and g > b
        green_mask = (g > r) & (g > b)
        return float(np.mean(green_mask))
    
    def _estimate_sky_brightness(self, image_array: np.ndarray) -> float:
        """Estimate brightness of the sky (upper portion of image)"""
        if len(image_array) == 0:
            return 0.5
        
        # Analyze top 30% of image (typically sky)
        sky_region = image_array[:int(image_array.shape[0]*0.3), :, :]
        brightness = float(np.mean(sky_region) / 255.0)
        return brightness
    
    def _estimate_overall_brightness(self, image_array: np.ndarray) -> float:
        """Estimate overall image brightness"""
        if len(image_array) == 0:
            return 0.5
        
        brightness = float(np.mean(image_array) / 255.0)
        return brightness
    
    def _has_distinctive_colors(self, image_array: np.ndarray) -> bool:
        """Check if image has distinctive/saturated colors"""
        if len(image_array) == 0:
            return False
        
        # High color saturation might indicate landmarks/distinct architecture
        saturation = float(np.std(image_array))
        return saturation > 50
    
    def _has_high_contrast(self, image_array: np.ndarray) -> bool:
        """Check if image has high contrast (urban indicator)"""
        if len(image_array) == 0:
            return False
        
        contrast = float(np.std(image_array) / np.mean(image_array + 1e-6))
        return contrast > 0.3
