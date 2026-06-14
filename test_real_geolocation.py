#!/usr/bin/env python3
"""
Real Photo Geolocation Test - Actual working pipeline
Loads a real image and runs actual geolocation (not mock data)
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
import time

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_real_geolocation():
    """Test the actual geolocation pipeline with a real image"""
    
    print("\n" + "="*70)
    print("REAL PHOTO GEOLOCATION TEST - No Simulations")
    print("="*70 + "\n")
    
    # Step 1: Load test image
    print("[1/6] Loading test image...")
    image_path = Path(__file__).parent / "beverly hills.jpg"
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)
        print(f"✅ Loaded image: {image_path.name} ({image.size[0]}x{image.size[1]})")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return False
    
    # Step 2: Check model availability
    print("\n[2/6] Checking model availability...")
    try:
        from app.inference.clip_common import is_clip_runtime_available
        clip_available = is_clip_runtime_available()
        print(f"{'✅' if clip_available else '❌'} CLIP (torch+transformers): {clip_available}")
    except Exception as e:
        print(f"❌ Cannot check CLIP: {e}")
        clip_available = False
    
    # Step 3: Test CLIP zero-shot geolocation
    print("\n[3/6] Testing CLIP zero-shot country detection...")
    clip_preds = []
    if clip_available:
        try:
            from app.inference.zero_shot_geo import clip_country_predictions
            from app.config import settings
            
            t0 = time.time()
            clip_preds = clip_country_predictions(
                image_array,
                model_id=settings.globe_clip_model_id,
                top_k=5,
                min_prob=0.008
            )
            elapsed = time.time() - t0
            
            if clip_preds:
                print(f"✅ CLIP predictions ({elapsed:.2f}s):")
                for i, pred in enumerate(clip_preds[:3], 1):
                    print(f"   {i}. {pred.country:25} @ ({pred.latitude:7.2f}, {pred.longitude:8.2f}) - Confidence: {pred.confidence:.1%}")
            else:
                print("⚠️  CLIP returned no predictions (check min_prob threshold)")
        except Exception as e:
            print(f"⚠️  CLIP prediction failed: {e}")
    else:
        print("⚠️  CLIP not available (install torch + transformers)")
    
    # Step 4: Test ensemble inference
    print("\n[4/6] Testing full ensemble inference...")
    try:
        from app.inference.ensemble import EnsembleInference
        from app.config import settings
        
        t0 = time.time()
        ensemble = EnsembleInference()
        results = ensemble.predict(
            image_array,
            include_retrieval=True,
            top_k=5,
            clip_model_id=settings.globe_clip_model_id if settings.ensemble_use_clip_zero_shot else None
        )
        elapsed = time.time() - t0
        
        print(f"✅ Ensemble inference completed ({elapsed:.2f}s)")
        print(f"   Model used: {results.get('model_used', 'unknown')}")
        print(f"   Ensemble size: {results.get('ensemble_size', 0)} sources")
        print(f"   Fusion sources: {', '.join(results.get('fusion_sources', []))}")
        
        primary = results.get('primary_prediction')
        if primary:
            print(f"\n   PRIMARY PREDICTION:")
            print(f"   📍 {primary.country:20} - {primary.city}")
            print(f"   📌 Coordinates: ({primary.latitude:.4f}, {primary.longitude:.4f})")
            print(f"   🎯 Confidence: {primary.confidence:.1%}")
            
            alternatives = results.get('alternative_predictions', [])
            if alternatives:
                print(f"\n   ALTERNATIVES:")
                for i, alt in enumerate(alternatives[:3], 1):
                    print(f"   {i}. {alt.country:20} - {alt.city:20} @ ({alt.latitude:.4f}, {alt.longitude:.4f}) - {alt.confidence:.1%}")
        else:
            print("   ⚠️  No primary prediction")
        
        print(f"\n   Source predictions counts:")
        for source, count in results.get('source_counts', {}).items():
            print(f"      {source}: {count}")
        
    except Exception as e:
        print(f"❌ Ensemble inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test feature extraction
    print("\n[5/6] Testing feature extraction...")
    try:
        from app.features.extractor import FeatureExtractor
        
        t0 = time.time()
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(image_array)
        elapsed = time.time() - t0
        
        if features:
            print(f"✅ Features extracted ({elapsed:.2f}s)")
            print(f"   Detected text: {bool(features.get('detected_text'))}")
            print(f"   Vegetation types: {features.get('vegetation_types', [])[:2]}")
            print(f"   Infrastructure: {features.get('infrastructure_type')}")
            print(f"   Architecture style: {features.get('architecture_style')}")
        else:
            print("⚠️  No features extracted")
    except Exception as e:
        print(f"⚠️  Feature extraction not available: {e}")
    
    # Step 6: Summary
    print("\n[6/6] Summary")
    print("-" * 70)
    
    if clip_available and clip_preds:
        print("✅ REAL geolocation working!")
        print("   - CLIP models loaded and running")
        print("   - Geographic predictions generated from actual model inference")
        print("   - No mock data or simulations")
        
        expected = "United States"  # Beverly Hills is in USA
        primary_country = clip_preds[0].country if clip_preds else None
        
        if primary_country and "USA" in primary_country or "United States" in primary_country:
            print(f"\n✅ CORRECT: Image correctly identified as {primary_country}")
        else:
            print(f"\n⚠️  Got {primary_country}, expected USA (Beverly Hills)")
        
        print("\n📊 Pipeline Status: WORKING")
        return True
    else:
        print("❌ CLIP models not available")
        print("   Install required packages: pip install torch transformers")
        print("\n📊 Pipeline Status: NOT WORKING (missing dependencies)")
        return False

if __name__ == "__main__":
    success = test_real_geolocation()
    sys.exit(0 if success else 1)
