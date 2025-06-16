#!/usr/bin/env python
"""
FastAPI Server for Emotion Recognition
Uses the trained model from emotion_system_paper_compliant.py
Processes angular features from Android client
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import joblib
import json
import httpx
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
model = None
scaler = None
pca = None
metadata = None

# Emotion mapping
EMOTION_MAPPING = {
    0: 'surprise',
    1: 'fear',
    2: 'disgust',
    3: 'happiness',
    4: 'sadness',
    5: 'anger',
    6: 'neutral'
}

# Recommendation service URL
RECOMMENDATION_SERVICE_URL = "http://localhost:8080"  # Update with actual URL


def load_model(model_path: str = "emotion_model_paper_compliant"):
    """Load the trained model and preprocessing components"""
    global model, scaler, pca, metadata

    model_path = Path(model_path)

    try:
        logger.info(f"Loading model from {model_path}")

        # Load model
        model_file = model_path / "model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model = joblib.load(model_file)
        logger.info(f"Loaded model: {type(model).__name__}")

        # Load scaler
        scaler_file = model_path / "scaler.pkl"
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        scaler = joblib.load(scaler_file)
        logger.info("Loaded StandardScaler")

        # Load PCA
        pca_file = model_path / "pca.pkl"
        if not pca_file.exists():
            raise FileNotFoundError(f"PCA file not found: {pca_file}")
        pca = joblib.load(pca_file)
        logger.info(f"Loaded PCA with {pca.n_components_} components")

        # Load metadata
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info("Loaded model metadata")
        else:
            logger.warning("Metadata file not found, using defaults")
            metadata = {
                "emotion_mapping": EMOTION_MAPPING,
                "angular_features": 10
            }

        logger.info("Model loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
    yield
    # Shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Emotion Recognition API",
    description="API for processing angular features from MediaPipe landmarks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class LandmarksRequest(BaseModel):
    """Request model for landmarks"""
    user_id: str
    content_id: str
    landmarks: List[List[float]]
    timestamp_ms: int


class AngularFeaturesRequest(BaseModel):
    """Request model for angular features"""
    angular_features: List[float]
    metadata: Dict[str, Any] = {}


class EmotionResponse(BaseModel):
    """Response model for emotion prediction"""
    emotion: str
    confidence: float
    emotion_scores: Dict[str, float]
    processing_time_ms: float
    metadata: Dict[str, Any]


class SimplifiedEmotionResponse(BaseModel):
    """Simplified response for Android client"""
    user_id: str
    content_id: str
    emotion: str
    confidence: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str


# EXACT PAPER SPECIFICATIONS from emotion_system_paper_compliant.py

# Table 1: Selected key landmarks (27 vertices)
SELECTED_LANDMARKS = {
    0: 61,  # Mouth end (right)
    1: 292,  # Mouth end (left)
    2: 0,  # Upper lip (middle)
    3: 17,  # Lower lip (middle)
    4: 50,  # Right cheek
    5: 280,  # Left cheek
    6: 48,  # Nose right end
    7: 4,  # Nose tip
    8: 289,  # Nose left end
    9: 206,  # Upper jaw (right)
    10: 426,  # Upper jaw (left)
    11: 133,  # Right eye (inner)
    12: 130,  # Right eye (outer)
    13: 159,  # Right upper eyelid (middle)
    14: 145,  # Right lower eyelid (middle)
    15: 362,  # Left eye (inner)
    16: 359,  # Left eye (outer)
    17: 386,  # Left upper eyelid (middle)
    18: 374,  # Left lower eyelid (middle)
    19: 122,  # Nose bridge (right)
    20: 351,  # Nose bridge (left)
    21: 46,  # Right eyebrow (outer)
    22: 105,  # Right eyebrow (middle)
    23: 107,  # Right eyebrow (inner)
    24: 276,  # Left eyebrow (outer)
    25: 334,  # Left eyebrow (middle)
    26: 336  # Left eyebrow (inner)
}

# Table 5: Angular features (10 angles)
ANGLE_TRIPLETS = [
    (2, 0, 3),  # θ1
    (0, 2, 1),  # θ2
    (6, 7, 8),  # θ3
    (9, 7, 10),  # θ4
    (0, 7, 1),  # θ5
    (1, 5, 8),  # θ6
    (1, 10, 8),  # θ7
    (13, 12, 14),  # θ8
    (21, 22, 23),  # θ9
    (6, 19, 23)  # θ10
]


def select_key_landmarks(all_landmarks: np.ndarray) -> np.ndarray:
    """
    Select 27 key landmarks from 468 MediaPipe landmarks (Table 1)

    Args:
        all_landmarks: Array of 468 landmarks from MediaPipe

    Returns:
        Array of 27 selected key landmarks
    """
    key_landmarks = []

    for vertex_id, mediapipe_id in SELECTED_LANDMARKS.items():
        if mediapipe_id < len(all_landmarks):
            landmark = all_landmarks[mediapipe_id]
            key_landmarks.append([landmark[0], landmark[1]])  # Use only x, y coordinates
        else:
            # Fallback if landmark index is out of range
            key_landmarks.append([0.0, 0.0])

    return np.array(key_landmarks)


def calculate_angle_between_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle θ between three points using exact paper formulas (Equations 1-3)

    Args:
        p1, p2, p3: Points as [x, y] coordinates

    Returns:
        Angle in degrees [0, 360]
    """
    # Equation 1: β = arctan((y3-y2)/(x3-x2))
    beta = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])

    # Equation 2: α = arctan((y1-y2)/(x1-x2))
    alpha = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])

    # Equation 3: θ = β - α
    theta = beta - alpha

    # Convert to degrees
    theta_degrees = np.degrees(theta)

    # Normalize to [0, 360] as per paper
    if theta_degrees < 0:
        theta_degrees += 360

    return theta_degrees


def compute_angular_features(landmarks: List[List[float]]) -> List[float]:
    """
    Compute angular features from facial landmarks using exact paper methodology
    """
    # Convert to numpy array (expecting 468 landmarks)
    all_landmarks = np.array(landmarks)

    if len(all_landmarks) != 468:
        logger.warning(f"Expected 468 landmarks, got {len(all_landmarks)}")
        # Pad or truncate to 468
        if len(all_landmarks) < 468:
            # Pad with zeros
            padding = np.zeros((468 - len(all_landmarks), 3))
            all_landmarks = np.vstack([all_landmarks, padding])
        else:
            # Truncate
            all_landmarks = all_landmarks[:468]

    # Select 27 key landmarks
    key_landmarks = select_key_landmarks(all_landmarks)

    # Extract 10 angular features
    angular_features = []

    for i, (v1, v2, v3) in enumerate(ANGLE_TRIPLETS):
        if v1 < len(key_landmarks) and v2 < len(key_landmarks) and v3 < len(key_landmarks):
            p1 = key_landmarks[v1]
            p2 = key_landmarks[v2]
            p3 = key_landmarks[v3]

            angle = calculate_angle_between_points(p1, p2, p3)
            angular_features.append(angle)
        else:
            # Fallback if indices are out of range
            angular_features.append(0.0)

    if len(angular_features) != 10:
        logger.error(f"Expected 10 angular features, got {len(angular_features)}")
        # Ensure we always return 10 features
        while len(angular_features) < 10:
            angular_features.append(0.0)
        angular_features = angular_features[:10]

    return angular_features


async def notify_recommendation_service(user_id: str, content_id: str, emotion: str, confidence: float):
    """Send emotion analysis results to recommendation service"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{RECOMMENDATION_SERVICE_URL}/recommendations/analyze",
                json={
                    "user_id": user_id,
                    "content_id": content_id,
                    "emotion": emotion,
                    "confidence": confidence
                },
                timeout=10.0
            )
            if response.status_code == 200:
                logger.info(f"Successfully notified recommendation service for content {content_id}")
            else:
                logger.error(f"Recommendation service returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Failed to notify recommendation service: {e}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="ok",
        model_loaded=(model is not None),
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=(model is not None),
        timestamp=datetime.now().isoformat()
    )


@app.post("/analyze_landmarks", response_model=SimplifiedEmotionResponse)
async def analyze_landmarks(request: LandmarksRequest):
    """
    Analyze emotion from facial landmarks

    Args:
        request: LandmarksRequest with user_id, content_id, landmarks, and timestamp

    Returns:
        SimplifiedEmotionResponse with predicted emotion and confidence
    """
    start_time = datetime.now()

    # Check if model is loaded
    if model is None or scaler is None or pca is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Extract angular features from landmarks
        angular_features = compute_angular_features(request.landmarks)

        if len(angular_features) != 10:
            raise ValueError(f"Expected 10 angular features, got {len(angular_features)}")

        # Convert to numpy array
        features = np.array(angular_features).reshape(1, -1)

        # Apply preprocessing
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # Get prediction
        prediction = model.predict(features_pca)[0]

        # Get probabilities if available
        emotion_scores = {}
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_pca)[0]
            for i, prob in enumerate(probabilities):
                if i < len(EMOTION_MAPPING):
                    emotion_scores[EMOTION_MAPPING[i]] = float(prob)
        else:
            # For models without predict_proba, set 1.0 for predicted class
            for i, emotion in EMOTION_MAPPING.items():
                emotion_scores[emotion] = 1.0 if i == prediction else 0.0

        # Get predicted emotion name
        predicted_emotion = EMOTION_MAPPING.get(prediction, 'unknown')
        confidence = emotion_scores.get(predicted_emotion, 0.0)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Processed emotion for content {request.content_id}: {predicted_emotion} (confidence: {confidence:.3f})")

        # Notify recommendation service asynchronously
        asyncio.create_task(notify_recommendation_service(
            request.user_id,
            request.content_id,
            predicted_emotion,
            confidence
        ))

        # Return simplified response for Android client
        return SimplifiedEmotionResponse(
            user_id=request.user_id,
            content_id=request.content_id,
            emotion=predicted_emotion,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Error processing landmarks: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/analyze_emotion", response_model=EmotionResponse)
async def analyze_emotion(request: AngularFeaturesRequest):
    """
    Analyze emotion from angular features (original endpoint)

    Args:
        request: AngularFeaturesRequest with 10 angular features

    Returns:
        EmotionResponse with predicted emotion and confidence
    """
    start_time = datetime.now()

    # Check if model is loaded
    if model is None or scaler is None or pca is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    if len(request.angular_features) != 10:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 10 angular features, got {len(request.angular_features)}"
        )

    try:
        # Convert to numpy array
        features = np.array(request.angular_features).reshape(1, -1)

        # Apply preprocessing
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # Get prediction
        prediction = model.predict(features_pca)[0]

        # Get probabilities if available
        emotion_scores = {}
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_pca)[0]
            for i, prob in enumerate(probabilities):
                if i < len(EMOTION_MAPPING):
                    emotion_scores[EMOTION_MAPPING[i]] = float(prob)
        else:
            # For models without predict_proba, set 1.0 for predicted class
            for i, emotion in EMOTION_MAPPING.items():
                emotion_scores[emotion] = 1.0 if i == prediction else 0.0

        # Get predicted emotion name
        predicted_emotion = EMOTION_MAPPING.get(prediction, 'unknown')
        confidence = emotion_scores.get(predicted_emotion, 0.0)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Prepare response
        response = EmotionResponse(
            emotion=predicted_emotion,
            confidence=confidence,
            emotion_scores=emotion_scores,
            processing_time_ms=processing_time,
            metadata={
                "model_type": type(model).__name__,
                "pca_components": int(pca.n_components_),
                "angular_features": request.angular_features,
                "client_metadata": request.metadata
            }
        )

        logger.info(f"Processed emotion: {predicted_emotion} (confidence: {confidence:.3f})")
        return response

    except Exception as e:
        logger.error(f"Error processing emotion: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")

    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "emotions": list(EMOTION_MAPPING.values()),
        "expected_features": 10,
        "feature_names": [f"θ{i + 1}" for i in range(10)],
        "preprocessing": {
            "scaler": "StandardScaler",
            "pca_components": pca.n_components_ if pca else None,
            "explained_variance": float(pca.explained_variance_ratio_.sum()) if pca else None
        },
        "metadata": metadata
    }


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn

    # Check if model exists
    model_path = Path("emotion_model_paper_compliant")
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        logger.info("Please train the model first using emotion_system_paper_compliant.py")
        exit(1)

    # Run server
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)