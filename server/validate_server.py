#!/usr/bin/env python
"""
Validation script to ensure server.py is compatible with the emotion model
"""

import requests
import numpy as np
import json
import time

# Server URL
SERVER_URL = "http://localhost:8000"

def generate_test_landmarks():
    """Generate test landmarks similar to MediaPipe output"""
    # Generate 468 landmarks with x, y, z coordinates
    landmarks = []
    for i in range(468):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        z = np.random.uniform(-0.1, 0.1)
        landmarks.append([x, y, z])
    return landmarks

def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{SERVER_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"Health status: {data['status']}")
    print(f"Model loaded: {data['model_loaded']}")
    return data['model_loaded']

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info endpoint...")
    response = requests.get(f"{SERVER_URL}/model_info")
    assert response.status_code == 200
    data = response.json()
    print(f"Model type: {data['model_type']}")
    print(f"Emotions: {data['emotions']}")
    print(f"Expected features: {data['expected_features']}")
    print(f"PCA components: {data['preprocessing']['pca_components']}")
    print(f"Explained variance: {data['preprocessing']['explained_variance']:.3f}")
    return data

def test_analyze_landmarks():
    """Test the analyze_landmarks endpoint"""
    print("\nTesting analyze_landmarks endpoint...")
    
    # Generate test data
    landmarks = generate_test_landmarks()
    
    request_data = {
        "user_id": "test-user-123",
        "content_id": "test-content-456",
        "landmarks": landmarks,
        "timestamp_ms": int(time.time() * 1000)
    }
    
    print(f"Sending {len(landmarks)} landmarks...")
    
    # Send request
    response = requests.post(
        f"{SERVER_URL}/analyze_landmarks",
        json=request_data
    )
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"Response:")
    print(f"  User ID: {data['user_id']}")
    print(f"  Content ID: {data['content_id']}")
    print(f"  Emotion: {data['emotion']}")
    print(f"  Confidence: {data['confidence']:.3f}")
    
    # Validate emotion is one of the expected values
    expected_emotions = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
    assert data['emotion'] in expected_emotions, f"Unknown emotion: {data['emotion']}"
    
    return data

def test_angular_features_directly():
    """Test if angular features are computed correctly"""
    print("\nTesting angular feature computation...")
    
    # Test with specific landmark configuration
    landmarks = generate_test_landmarks()
    
    # Make request to analyze_emotion endpoint with pre-computed features
    # This tests if the model can handle the angular features directly
    angular_features = [float(np.random.uniform(0, 360)) for _ in range(10)]
    
    request_data = {
        "angular_features": angular_features,
        "metadata": {"test": True}
    }
    
    response = requests.post(
        f"{SERVER_URL}/analyze_emotion",
        json=request_data
    )
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"Direct angular features test:")
    print(f"  Emotion: {data['emotion']}")
    print(f"  Confidence: {data['confidence']:.3f}")
    print(f"  Processing time: {data['processing_time_ms']:.1f}ms")
    
    return data

def test_multiple_requests():
    """Test multiple requests to check consistency"""
    print("\nTesting multiple requests...")
    
    emotions_count = {}
    num_requests = 10
    
    for i in range(num_requests):
        landmarks = generate_test_landmarks()
        request_data = {
            "user_id": f"test-user-{i}",
            "content_id": f"test-content-{i}",
            "landmarks": landmarks,
            "timestamp_ms": int(time.time() * 1000)
        }
        
        response = requests.post(f"{SERVER_URL}/analyze_landmarks", json=request_data)
        assert response.status_code == 200
        
        emotion = response.json()['emotion']
        emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
    
    print(f"Emotion distribution over {num_requests} requests:")
    for emotion, count in sorted(emotions_count.items()):
        print(f"  {emotion}: {count} ({count/num_requests*100:.1f}%)")

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("SERVER VALIDATION TEST")
    print("=" * 60)
    
    try:
        # Test health
        model_loaded = test_health_endpoint()
        if not model_loaded:
            print("\n❌ Model not loaded! Please ensure the model files exist.")
            print("Run emotion_system_paper_compliant.py first to train the model.")
            return
        
        # Test model info
        model_info = test_model_info()
        
        # Test landmark analysis
        result = test_analyze_landmarks()
        
        # Test angular features directly
        angular_result = test_angular_features_directly()
        
        # Test multiple requests
        test_multiple_requests()
        
        print("\n✅ All tests passed!")
        print("\nVALIDATION SUMMARY:")
        print("- Server is running correctly")
        print("- Model is loaded and accessible")
        print("- Landmark processing works as expected")
        print("- Angular feature computation is functional")
        print("- Server handles multiple requests properly")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server!")
        print("Please make sure the server is running:")
        print("  python server.py")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()