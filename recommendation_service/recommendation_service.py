#!/usr/bin/env python
"""
Recommendation Service
Manages content recommendations based on user emotions and preferences
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
import uuid
import logging
import random
import os
from bson import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Recommendation Service API",
    description="Content recommendation based on emotional feedback",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://RootUser:1q2w3e4r@furniturestore.oc1kln5.mongodb.net/?retryWrites=true&w=majority&appName=FurnitureStore")
DATABASE_NAME = "recommendation_service"

# Global MongoDB client
motor_client = None
db = None


# Models
class FeedItem(BaseModel):
    content_id: str
    type: str  # "photo" or "video"
    thumbnail_url: str
    media_url: str
    category: Optional[str] = None


class LikeRequest(BaseModel):
    user_id: str
    content_id: str


class EmotionAnalysisRequest(BaseModel):
    user_id: str
    content_id: str
    emotion: str
    confidence: float


class SuccessResponse(BaseModel):
    success: bool
    message: Optional[str] = None


# Emotion to preference mapping
EMOTION_PREFERENCE_MAPPING = {
    'happiness': 1.0,
    'surprise': 0.5,
    'neutral': 0.0,
    'fear': -0.5,
    'sadness': -0.5,
    'anger': -0.5,
    'disgust': -1.0
}


# MongoDB Collections
def get_collections():
    """Get MongoDB collections"""
    return {
        'content': db.content,
        'users': db.users,
        'user_preferences': db.user_preferences,
        'likes': db.likes,
        'emotion_logs': db.emotion_logs
    }


async def init_database():
    """Initialize MongoDB collections and indexes"""
    collections = get_collections()

    # Create indexes
    await collections['content'].create_index([("content_id", ASCENDING)], unique=True)
    await collections['content'].create_index([("category", ASCENDING)])
    await collections['content'].create_index([("created_at", DESCENDING)])

    await collections['users'].create_index([("user_id", ASCENDING)], unique=True)

    await collections['user_preferences'].create_index([("user_id", ASCENDING), ("category", ASCENDING)], unique=True)

    await collections['likes'].create_index([("user_id", ASCENDING), ("content_id", ASCENDING)], unique=True)
    await collections['likes'].create_index([("content_id", ASCENDING)])

    await collections['emotion_logs'].create_index([("user_id", ASCENDING)])
    await collections['emotion_logs'].create_index([("content_id", ASCENDING)])
    await collections['emotion_logs'].create_index([("created_at", DESCENDING)])

    # Check if content collection is empty and insert sample data
    count = await collections['content'].count_documents({})
    if count == 0:
        await insert_sample_content()

    logger.info("MongoDB initialized successfully")


async def insert_sample_content():
    """Insert sample content for testing"""
    categories = ['happiness', 'surprise', 'neutral', 'sadness', 'anger', 'fear', 'disgust']
    collections = get_collections()

    sample_content = []
    for i in range(100):  # Create 100 sample items
        content_type = random.choice(['photo', 'video'])
        category = random.choice(categories)
        content_id = str(uuid.uuid4())

        sample_content.append({
            "content_id": content_id,
            "type": content_type,
            "media_url": f"https://picsum.photos/id/{i % 100}/400/300",
            "thumbnail_url": f"https://picsum.photos/id/{i % 100}/200/150",
            "category": category,
            "created_at": datetime.utcnow()
        })

    await collections['content'].insert_many(sample_content)
    logger.info(f"Inserted {len(sample_content)} sample content items")


async def get_or_create_user(user_id: str) -> bool:
    """Get or create a user record, returns True if new user"""
    collections = get_collections()

    # Try to insert new user
    try:
        await collections['users'].insert_one({
            "user_id": user_id,
            "created_at": datetime.utcnow()
        })

        # Initialize preferences for all categories
        categories = ['happiness', 'surprise', 'neutral', 'sadness', 'anger', 'fear', 'disgust']
        preferences = []
        for category in categories:
            preferences.append({
                "user_id": user_id,
                "category": category,
                "score": 0.0,
                "updated_at": datetime.utcnow()
            })

        await collections['user_preferences'].insert_many(preferences)
        logger.info(f"Created new user: {user_id}")
        return True

    except Exception as e:
        # User already exists
        return False


async def get_user_preferences(user_id: str) -> Dict[str, float]:
    """Get user preference scores by category"""
    collections = get_collections()

    cursor = collections['user_preferences'].find({"user_id": user_id})
    preferences = {}
    async for doc in cursor:
        preferences[doc['category']] = doc['score']

    return preferences


async def update_user_preference(user_id: str, category: str, delta: float):
    """Update user preference score for a category"""
    collections = get_collections()

    await collections['user_preferences'].update_one(
        {"user_id": user_id, "category": category},
        {
            "$inc": {"score": delta},
            "$set": {"updated_at": datetime.utcnow()}
        },
        upsert=True
    )


@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection on startup"""
    global motor_client, db

    motor_client = AsyncIOMotorClient(MONGODB_URL)
    db = motor_client[DATABASE_NAME]

    await init_database()
    logger.info("Connected to MongoDB")


@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on shutdown"""
    motor_client.close()
    logger.info("Disconnected from MongoDB")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Recommendation Service is running", "database": "MongoDB"}


@app.get("/recommendations", response_model=List[FeedItem])
async def get_recommendations(user_id: str):
    """
    Get content recommendations for a user

    For new users: return diverse content from all categories
    For existing users: return content based on preferences
    """
    collections = get_collections()
    is_new_user = await get_or_create_user(user_id)

    if is_new_user:
        # New user: return diverse content
        # Get some items from each category
        pipeline = [
            {"$match": {"category": {"$ne": None}}},
            {"$group": {
                "_id": "$category",
                "items": {"$push": "$$ROOT"}
            }},
            {"$project": {
                "items": {"$slice": ["$items", 3]}  # 3 items per category
            }},
            {"$unwind": "$items"},
            {"$replaceRoot": {"newRoot": "$items"}},
            {"$limit": 30}
        ]

        cursor = collections['content'].aggregate(pipeline)
        content_items = await cursor.to_list(length=30)

        # If not enough diverse content, add random items
        if len(content_items) < 30:
            additional_items = await collections['content'].find(
                {"content_id": {"$nin": [item['content_id'] for item in content_items]}}
            ).limit(30 - len(content_items)).to_list(length=30)
            content_items.extend(additional_items)

    else:
        # Existing user: get preferences and recommend accordingly
        preferences = await get_user_preferences(user_id)

        # Sort categories by preference score
        sorted_categories = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
        top_categories = [cat for cat, score in sorted_categories[:3] if score > 0]

        if not top_categories:
            # If no positive preferences, return diverse content
            content_items = await collections['content'].find(
                {"category": {"$ne": None}}
            ).limit(30).to_list(length=30)
        else:
            # Get 70% from top categories, 30% from others
            top_items = await collections['content'].find(
                {"category": {"$in": top_categories}}
            ).limit(21).to_list(length=21)

            other_items = await collections['content'].find(
                {"category": {"$nin": top_categories}}
            ).limit(9).to_list(length=9)

            content_items = top_items + other_items
            random.shuffle(content_items)

    # Convert to FeedItem objects
    feed_items = []
    for item in content_items:
        feed_items.append(FeedItem(
            content_id=item['content_id'],
            type=item['type'],
            thumbnail_url=item['thumbnail_url'],
            media_url=item['media_url'],
            category=item.get('category')
        ))

    logger.info(f"Returning {len(feed_items)} items for user {user_id}")
    return feed_items


@app.post("/recommendations/{content_id}/like", response_model=SuccessResponse)
async def like_content(content_id: str, request: LikeRequest):
    """
    Record a like for content and update user preferences
    """
    collections = get_collections()

    try:
        # Check if content exists
        content = await collections['content'].find_one({"content_id": content_id})
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Try to insert like (will fail if already exists due to unique index)
        try:
            await collections['likes'].insert_one({
                "user_id": request.user_id,
                "content_id": content_id,
                "created_at": datetime.utcnow()
            })
        except Exception:
            return SuccessResponse(success=False, message="Already liked")

        # Update user preferences if category is known
        if content.get('category'):
            await update_user_preference(request.user_id, content['category'], 1.0)

        logger.info(f"User {request.user_id} liked content {content_id}")
        return SuccessResponse(success=True, message="Like recorded")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording like: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendations/analyze", response_model=SuccessResponse)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """
    Process emotion analysis results and update user preferences
    """
    collections = get_collections()

    try:
        # Log emotion analysis
        await collections['emotion_logs'].insert_one({
            "user_id": request.user_id,
            "content_id": request.content_id,
            "emotion": request.emotion,
            "confidence": request.confidence,
            "created_at": datetime.utcnow()
        })

        # Get content to check category
        content = await collections['content'].find_one({"content_id": request.content_id})

        if content:
            category = content.get('category')

            # If category is not set, try to infer it from emotions
            if not category:
                # Simple heuristic: if many users feel the same emotion for this content,
                # set that as the category
                emotion_count = await collections['emotion_logs'].count_documents({
                    "content_id": request.content_id,
                    "emotion": request.emotion
                })

                if emotion_count >= 5:  # Threshold for category assignment
                    category = request.emotion
                    await collections['content'].update_one(
                        {"content_id": request.content_id},
                        {"$set": {"category": category}}
                    )
                    logger.info(f"Assigned category '{category}' to content {request.content_id}")

            # Update user preferences based on emotion
            if category:
                preference_delta = EMOTION_PREFERENCE_MAPPING.get(request.emotion, 0.0)
                # Weight by confidence
                weighted_delta = preference_delta * request.confidence

                await update_user_preference(request.user_id, category, weighted_delta)

                logger.info(f"Updated preferences for user {request.user_id}: "
                            f"{category} += {weighted_delta}")

        return SuccessResponse(success=True, message="Emotion analysis processed")

    except Exception as e:
        logger.error(f"Error processing emotion analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/{user_id}")
async def get_user_stats(user_id: str):
    """Get statistics for a user (for debugging/monitoring)"""
    collections = get_collections()

    # Get user preferences
    preferences = await get_user_preferences(user_id)

    # Get like count
    like_count = await collections['likes'].count_documents({"user_id": user_id})

    # Get emotion analysis count
    emotion_count = await collections['emotion_logs'].count_documents({"user_id": user_id})

    # Get emotion distribution
    emotion_pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {
            "_id": "$emotion",
            "count": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence"}
        }}
    ]

    emotion_stats = await collections['emotion_logs'].aggregate(emotion_pipeline).to_list(length=10)

    return {
        "user_id": user_id,
        "preferences": preferences,
        "total_likes": like_count,
        "total_emotions_analyzed": emotion_count,
        "emotion_distribution": {
            stat['_id']: {
                "count": stat['count'],
                "avg_confidence": stat['avg_confidence']
            }
            for stat in emotion_stats
        }
    }


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Recommendation Service on http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)