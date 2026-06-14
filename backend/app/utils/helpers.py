"""Helper utilities for image processing and EXIF handling"""
import logging
from typing import Optional, Dict, Tuple
from io import BytesIO
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _ratio_to_float(value) -> float:
    """Convert PIL rational / tuple / float to float."""
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            d = float(value.denominator)
            return float(value.numerator) / d if d else 0.0
        if isinstance(value, (tuple, list)) and len(value) == 2:
            d = float(value[1])
            return float(value[0]) / d if d else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _dms_tuple_to_decimal(dms, ref) -> float:
    """Convert EXIF GPS DMS tuple + ref (N/S/E/W) to signed decimal degrees."""
    if not dms or len(dms) < 3:
        raise ValueError("invalid GPS DMS")
    degrees = _ratio_to_float(dms[0])
    minutes = _ratio_to_float(dms[1])
    seconds = _ratio_to_float(dms[2])
    dec = degrees + minutes / 60.0 + seconds / 3600.0
    ref_s = ref.decode("ascii", errors="ignore") if isinstance(ref, bytes) else str(ref or "")
    ref_s = ref_s.upper()
    if ref_s in ("S", "W"):
        dec = -dec
    return dec


def extract_exif_gps(image_data: bytes) -> Optional[Dict[str, float]]:
    """
    Extract GPS coordinates from image EXIF (JPEG/TIFF) using Pillow.

    Returns dict with 'latitude' and 'longitude' or None if absent/invalid.
    """
    try:
        from PIL.ExifTags import IFD

        image = Image.open(BytesIO(image_data))
        exif = image.getexif()
        if not exif:
            return None

        gps_ifd = exif.get_ifd(IFD.GPS)
        if not gps_ifd:
            return None

        lat_dms = gps_ifd.get(2)
        lon_dms = gps_ifd.get(4)
        lat_ref = gps_ifd.get(1)
        lon_ref = gps_ifd.get(3)

        if lat_dms is None or lon_dms is None or lat_ref is None or lon_ref is None:
            return None

        latitude = _dms_tuple_to_decimal(lat_dms, lat_ref)
        longitude = _dms_tuple_to_decimal(lon_dms, lon_ref)

        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            return None

        return {"latitude": latitude, "longitude": longitude}

    except Exception as e:
        logger.debug(f"No EXIF GPS found: {e}")
        return None


def _dms_to_decimal(dms_tuple: Tuple) -> float:
    """Convert DMS (degrees, minutes, seconds) to decimal coordinates"""
    degrees = dms_tuple[0][0] / dms_tuple[0][1]
    minutes = dms_tuple[1][0] / dms_tuple[1][1] / 60.0
    seconds = dms_tuple[2][0] / dms_tuple[2][1] / 3600.0
    return degrees + minutes + seconds


def convert_to_numpy(image_data: bytes) -> Optional[np.ndarray]:
    """
    Convert image bytes to numpy array.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Numpy array (H, W, 3) or None if error
    """
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Error converting image to numpy: {e}")
        return None


def resize_image(image_array: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to standard size for models"""
    image = Image.fromarray(image_array)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(resized)


def normalize_image(image_array: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Normalize image pixel values to [-1, 1] or [0, 1]"""
    normalized = (image_array.astype(np.float32) / 255.0 - mean) / std
    return normalized


def blur_faces_and_plates(image_array: np.ndarray) -> np.ndarray:
    """
    Blur faces and license plates for privacy.
    
    Would use: OpenCV face detector + ANPR detector
    For now: return original (implement with real models later)
    """
    # In real implementation, would use:
    # - cv2.CascadeClassifier for faces
    # - ANPR/plate detection model
    return image_array


def validate_image_size(image_array: np.ndarray, max_size: int = 4096) -> bool:
    """Validate image dimensions are within acceptable range"""
    h, w = image_array.shape[:2]
    return h <= max_size and w <= max_size and h > 10 and w > 10


def reverse_geocode(latitude: float, longitude: float) -> Optional[Dict[str, str]]:
    """
    Reverse geocode coordinates to place name.

    Wire Google Maps, Nominatim, or another provider before returning structured data.
    Returns None until configured — no placeholder country/city strings.
    """
    logger.debug("reverse_geocode not configured (%s, %s)", latitude, longitude)
    return None
