"""Database models for storing geolocation results"""
from datetime import datetime
from typing import Optional


class GeoLocationResult:
    """In-memory storage for geolocation results (can be extended to SQL database)"""
    
    def __init__(
        self,
        image_id: str,
        latitude: float,
        longitude: float,
        country: str,
        city: Optional[str],
        confidence: float,
        processing_time_ms: float,
        model_used: str,
        has_exif_gps: bool = False,
    ):
        self.image_id = image_id
        self.latitude = latitude
        self.longitude = longitude
        self.country = country
        self.city = city
        self.confidence = confidence
        self.processing_time_ms = processing_time_ms
        self.model_used = model_used
        self.has_exif_gps = has_exif_gps
        self.created_at = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "image_id": self.image_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country": self.country,
            "city": self.city,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "has_exif_gps": self.has_exif_gps,
            "created_at": self.created_at.isoformat(),
        }


# In-memory database storage
results_store: dict = {}
