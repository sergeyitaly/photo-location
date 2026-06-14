# Photo Geolocation System - FastAPI Multimodule Solution

A production-ready Python FastAPI backend with modern web interface for AI-powered photo geolocation. Uses hybrid on-device and cloud inference with multiple ML models in an ensemble architecture.

## 🎯 Features

- **Multi-Model Ensemble**: CNN classifier + image retrieval + optional LVLM reasoning
- **50+ Visual Feature Extraction**: Landmarks, vegetation, architecture, weather, text, etc.
- **Hybrid Architecture**: On-device inference for privacy + cloud backend for heavy models
- **REST API**: Full-featured FastAPI with Swagger/ReDoc documentation
- **Web UI**: Modern responsive frontend with drag-drop image upload
- **Modular Design**: Clean separation of concerns for features, inference, and API layers
- **Privacy-First**: Optional on-device processing with explicit user consent
- **Production-Ready**: Includes Docker, CI/CD preparation, and monitoring hooks

## 📦 Project Structure

```
photo-geolocation/
├── backend/                          # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration & settings
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py           # Pydantic schemas for API
│   │   │   └── database.py          # Result storage
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   └── extractor.py         # 50+ feature extraction pipeline
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py        # CNN classification module
│   │   │   ├── retrieval.py         # Image retrieval (CLIP-based)
│   │   │   └── ensemble.py          # Multi-model ensemble
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py            # API endpoints
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py           # Image processing, EXIF, geocoding
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/                         # Web UI
│   ├── index.html                   # Main page
│   ├── styles.css                   # Styling (mobile-responsive)
│   └── app.js                       # Client-side logic
├── docker-compose.yml               # Docker Compose configuration
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- 2GB free disk space for dependencies

### Option 1: Local Development

1. **Clone/Setup repository**
   ```bash
   cd /Volumes/Data/PHOTO_LOCATION
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python3 -m venv venv
   source .venv312/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Run backend**
   ```bash
   cd backend
   python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open frontend**
   - **Web UI:** http://localhost:8000/ (or http://localhost:8000/static/index.html)
   - Swagger Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - Do **not** open `frontend/index.html` as a `file://` URL — the script and API paths will not load. Use the URLs above while uvicorn is running.

### Option 2: Docker Compose

```bash
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000 (if using nginx service)

## 📡 API Endpoints

### Core Endpoints

#### POST `/predict`
**Predict geolocation from a single image**

Request:
```json
{
  "base64_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "use_cloud_inference": false,
  "include_feature_analysis": true
}
```

Response:
```json
{
  "status": "success",
  "image_id": "img_a1b2c3d4",
  "primary_prediction": {
    "latitude": 48.8584,
    "longitude": 2.2945,
    "country": "France",
    "city": "Paris",
    "confidence": 0.92,
    "distance_confidence_km": 5.0
  },
  "alternative_predictions": [...],
  "feature_analysis": {
    "landmarks": [{"name": "Eiffel Tower", "confidence": 0.95}],
    "vegetation_types": ["urban"],
    "weather_condition": "sunny",
    ...
  },
  "processing_time_ms": 234.5,
  "model_used": "ensemble",
  "has_exif_gps": false
}
```

#### POST `/predict_batch`
**Batch predict multiple images**

Form data: Multiple image files

Response:
```json
{
  "results": [
    {
      "filename": "photo1.jpg",
      "status": "success",
      "prediction": {...},
      "processing_time_ms": 234.5
    }
  ],
  "total": 1
}
```

#### GET `/results/{image_id}`
**Retrieve previously computed result**

#### GET `/results`
**List all stored results** (with pagination)

Parameters:
- `limit` (int, default=100): Number of results to return
- `skip` (int, default=0): Number of results to skip

#### GET `/models/info`
**Get information about loaded models**

Response:
```json
{
  "ensemble_type": "Hybrid (CNN + Retrieval)",
  "classifier_info": {...},
  "retrieval_info": {...},
  "merge_strategy": "Weighted average by location cluster"
}
```

#### GET `/config`
**Get current configuration**

#### GET `/health`
**Health check endpoint**

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧠 Model Architecture

### 1. CNN Classifier (PlaNet-style)
- Hierarchical spatial classification
- Predicts: Country → City
- Fast, suitable for on-device
- ~50MB TensorFlow Lite variant

### 2. Image Retrieval (GeoCLIP-inspired)
- CLIP-based image-to-GPS embeddings
- Queries against vector database (FAISS)
- Fine-grained location prediction
- ~100MB model size

### 3. Ensemble Merger
- Combines predictions from multiple models
- Weighted by confidence and inter-model agreement
- Returns primary + alternative predictions

### Optional: LVLM Reasoning
- Uses GPT-4V or Gemini for chain-of-thought
- Analyzes complex scene understanding
- Higher accuracy but requires API call
- Cloud-only (privacy consideration)

## Techniques Implemented For Photo Geolocation

This project combines several complementary techniques to estimate location from a single photo.
Some of them directly affect the predicted coordinates, while others validate, rerank, or explain
the result.

### Techniques that directly influence the predicted location

- **Multi-resolution grid search + ensemble fusion**: the backend can now run a coarse-to-fine
  StreetCLIP grid search over the gazetteer, score winning coarse cells, refine into smaller cells,
  and then score real city labels only inside the best fine cells before fusing those results with
  the other geolocation sources.
- **EXIF GPS extraction**: if an image already contains embedded GPS metadata, the backend can use
  it as the primary location source.
- **Filename-based location hints**: a lightweight filename matcher can turn known place names in
  the uploaded filename into a location hypothesis.
- **GeoCLIP coordinate retrieval**: GeoCLIP produces coordinate candidates directly from image
  content.
- **StreetCLIP over a GeoNames gazetteer**: StreetCLIP scores the photo against a large city/town
  label set built from GeoNames and returns coordinate candidates for the best-matching places.
- **CLIP zero-shot country and landmark priors**: CLIP-based country and landmark predictions
  generate additional geographic hypotheses used by the fusion stage.
- **Weighted multi-model fusion**: the system merges CLIP zero-shot, GeoCLIP, and StreetCLIP
  candidates into a ranked list of predicted coordinates.
- **Hybrid GeoCLIP/StreetCLIP reconciliation**: after fusion, an alternate GeoCLIP rank can be
  promoted when it agrees better with the top StreetCLIP city hypothesis (same rules worldwide —
  no hardcoded city or country lists).
- **Open-water fusion hint (global)**: when generic blue-water pixels are visible in the frame,
  StreetCLIP’s fusion weight gets a small boost so gazetteer city names can compete with raw GPS
  clusters on lakes and coasts anywhere — this does **not** force a specific place name.

### Techniques that validate or refine a predicted pin

- **Wikipedia geosearch validation**: candidate coordinates can be cross-checked against nearby
  geo-tagged English Wikipedia articles.
- **OpenTopoData terrain validation**: local elevation and relief checks help reject pins that do
  not fit the physical setting implied by the photo.
- **Wikipedia semantic matching**: when enabled, CLIP image-text similarity can compare the photo
  against nearby Wikipedia lead-text extracts as an additional consistency signal.
- **OpenStreetMap Nominatim reverse geocoding**: the final coordinates can be resolved into a human
  readable locality, region, county, and country name.
- **Local cross-reference database integration**: fused candidates can be cross-checked against the
  local gazetteer database using nearby-place lookup, city-name agreement, country agreement, and
  filename/place overlap before open-data validation.

### Techniques that provide supporting visual evidence

- **Globe regional CLIP cues**: macro-region prompt banks provide broad regional evidence without
  directly moving the pin.
- **Scene geolocation cues**: pixel statistics plus optional CLIP prompt banks describe vegetation,
  built environment, palette, climate/light, and maintenance-style cues.
- **ML recognition prompt banks**: CLIP prompt banks can surface scene/object labels that explain
  why certain locations are being favored.
- **Infrastructure and energy cues**: extra CLIP prompt banks look for visual hints related to
  electrical grid, gas, solar, wind, and other infrastructure.
- **Reading axes / interpretive summaries**: the backend generates short descriptive summaries for
  view geometry, built form, and Wikipedia-based plausibility to help interpret results.

### Techniques used to keep inference practical

- **StreetCLIP gazetteer geo-filtering**: when GeoCLIP has a prior, cities are scored inside a
  tight bbox (default ±2° lat / ±2.5° lon, expanded only when top GeoCLIP ranks disagree or too
  few settlements match). Progress UI shows the actual box, not a coarse ±10° region.
- **Gazetteer trimming by distance**: inside the bbox, nearest settlements to the GeoCLIP prior are
  preferred over population-only ranking so the final pin is city-scale, not continent-scale.
- **StreetCLIP early stop**: cities are scored country-by-country in batches; if a batch peaks below
  earlier batches (wrong direction), remaining labels are skipped and the best-so-far cities are kept.
  The progress UI will not show declining confidence % on leading guesses.
- **Outbound API throttling**: Wikipedia, Wikimedia Commons, and OpenTopoData requests are spaced,
  retried on HTTP 429 with backoff, cached by coordinate, and limited to the top few fusion candidates
  so public APIs are not hammered (OpenTopo free tier ≈1 request/second).
- **Prior-aware grid search**: with a GeoCLIP prior, map-grid cells use ~1.5° coarse and ~0.35° fine
  steps (~40 km) instead of 30° worldwide cells.
- **Model warmup at startup**: CLIP, GeoCLIP, and StreetCLIP weights can be loaded before the first
  prediction to reduce cold-start latency.
- **In-memory gazetteer caching**: the parsed StreetCLIP gazetteer stays cached in memory instead of
  being rebuilt from JSON on every request.
- **Fast prediction mode**: the API can skip slower enrichment steps such as external validation,
  reverse geocoding, and diagnostic CLIP panels when a quicker answer is preferred.

## 🎨 Visual Feature Pipeline

The system extracts 50+ visual features:

**Natural Environment**
- Tree/plant species (hemlock, palm, pine, etc.)
- Grassland condition (green vs dry)
- Soil/earth color (red laterite, black volcanic, etc.)
- Mountain shapes (Alps vs Rockies vs Himalayas)
- Coastline type (rocky cliff vs sandy beach)
- Water type/color

**Weather & Astronomy**
- Sun position & shadow direction
- Time-of-day light (golden hour, blue hour)
- Weather condition (clear, cloudy, rainy, foggy, snowy)
- Moon/stars (celestial matching)
- Sky color and hue
- Seasonal vegetation

**Infrastructure**
- Road surface quality & markings
- Lane marking style (double yellow vs white)
- Driving side (left vs right)
- Utility poles & power lines
- Guardrail style
- Manhole covers

**Architecture**
- Building roof style (gabled, slate, flat, thatched)
- Wall material/color
- Window & facade style
- Building density
- Historical architecture styles

**Cultural Indicators**
- Text/language on signs (Latin, Cyrillic, Arabic, etc.)
- Brand logos
- Clothing styles (coats vs shorts)
- Uniforms (school, military)
- Headwear (fez, ushanka, sombrero)
- Crowd behavior

**Vehicles & Transport**
- Vehicle types/models (Kei-cars, Tata Nano, Ladas)
- License plate shape/color
- Bicycle type
- Transit infrastructure

**Landmarks**
- Famous landmarks (Eiffel Tower, Statue of Liberty, etc.)
- Google Landmarks v2 detection (30k landmarks)

## ⚙️ Configuration

Edit `backend/.env` or `backend/.env.example`:

```bash
# App Settings
DEBUG=True
HOST=0.0.0.0
PORT=8000

# ML Model Settings
MODEL_DEVICE=cpu  # cpu or cuda
USE_ON_DEVICE_MODEL=True

# Feature Extraction Settings
ENABLE_LANDMARK_DETECTION=True
ENABLE_VEGETATION_ANALYSIS=True
ENABLE_ARCHITECTURE_ANALYSIS=True
ENABLE_TEXT_OCR=True

# Inference Settings
CONFIDENCE_THRESHOLD=0.3
TOP_K_PREDICTIONS=5

# Database
DATABASE_URL=sqlite:///./geolocations.db

# APIs (Optional)
GOOGLE_MAPS_API_KEY=your_key_here
```

## 🔐 Privacy & Security

- **On-Device Processing**: Core models run locally without data leaving device
- **Explicit Consent**: Users grant permission before any processing
- **No Data Storage**: Raw images not persisted (except transient processing)
- **HTTPS Ready**: All API calls support encryption
- **EXIF Stripping**: Optional privacy mode removes metadata
- **Face/Plate Blurring**: Available preprocessing step
- **Differential Privacy**: Can be added to training pipeline


## 🧪 Testing

```bash
cd backend

# Run unit tests
pytest

# Run with coverage
pytest --cov=app

# Test API with httpx client
python -c "import httpx; r = httpx.get('http://localhost:8000/health'); print(r.json())"
```

## 📊 Performance

### Inference Speed (on CPU)
- CNN Classification: ~50-100ms
- Image Retrieval: ~100-200ms
- Full Ensemble: ~150-300ms
- Processing on GPU: ~30-50ms faster

### Model Sizes
- MobileNet Classifier: 15-30MB
- CLIP Retrieval: 100MB
- Full Stack on-device: ~200MB
- Vector DB (FAISS): ~1-10GB (depends on index size)

### Accuracy improvements (vision path, no paid APIs)

These run automatically on every `/predict` (see `backend/app/config.py`):

| Feature | What it does |
|---------|----------------|
| **CLIP country filter** | Top 1–3 CLIP country softmax hypotheses restrict the StreetCLIP gazetteer (fewer wrong-continent cities). |
| **Dynamic fusion** | When StreetCLIP top1−top2 margin is strong, GeoCLIP weight is reduced; when GeoCLIP ranks scatter, GeoCLIP is down-weighted. |
| **CLIP landmarks in fusion** | Extra landmark prompts (Eiffel Tower, Brandenburg Gate, …) merged into CLIP branch even in fast mode. |
| **Fast gated grid** | In fast mode, a **small** coarse→fine grid runs only when GeoCLIP or CLIP country confidence is weak. |
| **Feature analysis** | Optional Tesseract OCR (`pytesseract` + system `tesseract`), CLIP architecture hint, CLIP landmark hints for Ollama/reasoning. |

Tuning (optional in `backend/.env`):

```env
STREETCLIP_COUNTRY_FILTER_ENABLED=True
STREETCLIP_COUNTRY_FILTER_MAX_COUNTRIES=3
FAST_MODE_CONFIDENCE_GATED_GRID=True
GEOCLIP_DOWNWEIGHT_WHEN_STREETCLIP_CONFIDENT=0.55
FEATURE_ANALYSIS_OCR_ENABLED=True
```

Optional OCR: `pip install pytesseract` and install [Tesseract](https://github.com/tesseract-ocr/tesseract) on your system.

### What to trust (vision pins)

| Scale | Typical usefulness | Verified by this app? |
|-------|-------------------|------------------------|
| **Region** (climate, relief, broad built form, infrastructure style) | Often directionally useful | Partially — soft cues only |
| **Country / macro-area** | Moderate when CLIP country + gazetteer agree | No — model estimate |
| **Exact village / street** | Frequently wrong on generic photos | **No** — treat as hypothesis |

For stronger verification, open the predicted coordinates in **satellite and street imagery** and compare: yellow gas pipes at roadsides, hillside orientation, road curvature, utility pole spacing, and house placement. The results UI includes links and a checklist; optional `GOOGLE_MAPS_API_KEY` enables automated Street View CLIP checks.

### Accuracy Benchmarks
- Country-scale signals: often plausible on distinctive scenes (not guaranteed)
- Named village / city from vision alone: **unverified** — errors of tens to hundreds of km are common
- Coordinate error: often 1–50 km on generic photos; landmarks can be much tighter
- EXIF GPS: use when present — strongest coordinate source in this app

## 🔄 Workflow

1. **User uploads photo** via web UI
2. **Frontend converts to base64** and sends to backend
3. **Backend extracts features** (landmarks, vegetation, text, etc.)
4. **ML models predict** country/city/coordinates:
   - CNN provides coarse classification
   - Retrieval provides fine-grained match
   - Ensemble merges results
5. **Feature analysis** displayed (optional)
6. **Results cached** and returned to frontend
7. **Frontend displays** predictions on map with confidence

## 🚢 Deployment

### Production Checklist
- [ ] Set `DEBUG=False` in `.env`
- [ ] Configure proper database (PostgreSQL recommended)
- [ ] Add authentication/rate limiting
- [ ] Enable HTTPS/TLS
- [ ] Set up proper logging/monitoring
- [ ] Configure CORS for your domain
- [ ] Add API key authentication
- [ ] Set up CI/CD pipeline
- [ ] Test on target hardware

### Docker Deployment
```bash
docker build -t photo-geo:latest backend/
docker run -p 8000:8000 -e DEBUG=False photo-geo:latest
```

### Cloud Deployment (AWS/GCP/Azure)
- Use AWS EC2/Lambda, GCP Cloud Run, or Azure Container Instances
- Store vector DB in managed service (AWS RDS, GCP Cloud SQL)
- Use CDN for frontend distribution
- Configure auto-scaling based on load

## 📈 Future Enhancements

### Phase 2 (MVP+)
- [ ] Real ML models (TensorFlow Lite, PyTorch)
- [ ] Vector database integration (FAISS, Weaviate)
- [ ] LVLM support (OpenAI API, Google Gemini)
- [ ] Photo sequence analysis (temporal coherence)
- [ ] Multi-image album processing
- [ ] User feedback loops for refinement

### Phase 3 (Production)
- [ ] Mobile app (Android via CameraX)
- [ ] Google Photos Picker integration
- [ ] Advanced privacy options
- [ ] Federated learning support
- [ ] Differential privacy in training
- [ ] Performance monitoring & telemetry

### Phase 4 (Advanced)
- [ ] LVLM chain-of-thought reasoning
- [ ] Custom model fine-tuning
- [ ] Offline landmark recognition
- [ ] Multi-modal inputs (photo + metadata)
- [ ] Real-time video processing
- [ ] 3D scene reconstruction

## 📚 References & Citations

Based on research papers:
- **PlaNet**: PlaNet - Photographic Location Estimation with Convolutional Neural Networks
- **GeoCLIP**: CLIP-based multi-modal retrieval for geo-localization
- **StreetCLIP**: Street View based contrastive learning
- **YFCC100M**: Large-scale geotagged image dataset
- **Google Landmarks v2**: 30k landmark recognition dataset
- **Im2GPS3k**: Image to GPS coordinate benchmark

## 📝 License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## 👥 Contributing

Pull requests welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

- **Issues**: Create GitHub issue
- **Docs**: See `/docs` endpoints or README
- **Examples**: See `frontend/app.js` for client usage

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Validation](https://docs.pydantic.dev/)
- [CLIP Model](https://openai.com/research/learning-transferable-models-for-computational-photography)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [OpenCV](https://opencv.org/)

---

Built with ❤️ for photographers and developers who want to know where their photos were taken.
