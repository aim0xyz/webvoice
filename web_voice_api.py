"""
Web Voice Recording API
Handles voice recordings from web invitations and submits them to Chatterbox API
"""
import os
import json
import base64
from typing import Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Firebase Admin SDK for Firestore access
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore_v1.base_query import FieldFilter
    
    # Initialize Firebase Admin SDK
    if not firebase_admin._apps:
        # Option 1: Try JSON string from environment variable (for free hosting)
        firebase_credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if firebase_credentials_json:
            try:
                cred = credentials.Certificate(json.loads(firebase_credentials_json))
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized from JSON environment variable")
            except Exception as e:
                print(f"⚠️ Error loading Firebase credentials from JSON: {e}")
                firebase_credentials_json = None
        
        # Option 2: Try credentials file path
        if not firebase_admin._apps:
            cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized from credentials file")
        
        # Option 3: Try default credentials (for cloud environments like GCP)
        if not firebase_admin._apps:
            try:
                firebase_admin.initialize_app()
                print("✅ Firebase initialized using default credentials")
            except Exception as e:
                print(f"⚠️ Could not initialize Firebase with default credentials: {e}")
    
    db = firestore.client()
    FIREBASE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Firebase Admin SDK not available: {e}")
    print("⚠️ Token validation will be disabled. Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
    db = None
    FIREBASE_AVAILABLE = False

# Chatterbox API configuration
# IMPORTANT: Set these as environment variables for security!
# DO NOT hardcode API keys in the file
CHATTERBOX_API_URL = os.environ.get(
    "CHATTERBOX_API_URL",
    "https://api.runpod.ai/v2/bxzprpjs1kpwpt/run"  # RunPod serverless endpoint
)
CHATTERBOX_API_KEY = os.environ.get(
    "CHATTERBOX_API_KEY",
    ""  # MUST be set via environment variable - never hardcode!
)

if not CHATTERBOX_API_KEY:
    print("⚠️ WARNING: CHATTERBOX_API_KEY not set. Voice cloning will fail.")
    print("⚠️ Please set CHATTERBOX_API_KEY environment variable in your hosting platform.")

# FastAPI app
app = FastAPI(title="Web Voice Recording API")

# CORS - Allow web app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your web domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_invitation_token(token: str) -> Optional[dict]:
    """
    Validate invitation token in Firestore and return invitation data
    Returns None if token is invalid or expired
    """
    if not FIREBASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured. Cannot validate invitation tokens."
        )
    
    try:
        doc_ref = db.collection("voice_invitations").document(token)
        doc = doc_ref.get()
        
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if not data:
            return None
        
        # Check status
        status = data.get("status", "pending")
        if status not in ["pending"]:
            return None  # Already completed, expired, or cancelled
        
        # Check expiration (Flutter stores as ISO string)
        expires_at = data.get("expiresAt")
        if expires_at:
            if isinstance(expires_at, datetime):
                expires = expires_at
            elif hasattr(expires_at, "to_datetime"):
                # Firestore Timestamp
                expires = expires_at.to_datetime()
            else:
                # ISO string (from Flutter)
                expires = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
            
            if datetime.now(expires.tzinfo if expires.tzinfo else None).replace(tzinfo=None) > expires.replace(tzinfo=None):
                # Mark as expired
                doc_ref.update({"status": "expired"})
                return None
        
        return data
    
    except Exception as e:
        print(f"Error validating invitation token: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating token: {str(e)}")


def update_invitation(token: str, voice_id: str, status: str = "completed"):
    """Update invitation in Firestore with voice ID and status"""
    if not FIREBASE_AVAILABLE:
        print("⚠️ Firebase not available, cannot update invitation")
        return
    
    try:
        doc_ref = db.collection("voice_invitations").document(token)
        doc_ref.update({
            "status": status,
            "completedAt": datetime.now().isoformat(),  # Convert to ISO string to match Flutter
            "voiceId": voice_id,  # Match Flutter field name (not recordedVoiceId)
        })
        print(f"✅ Updated invitation {token} with voice_id: {voice_id}")
    except Exception as e:
        print(f"⚠️ Error updating invitation: {e}")


async def submit_to_chatterbox(audio_bytes: bytes, user_id: str, voice_name: str) -> dict:
    """
    Submit audio to Chatterbox API for voice cloning
    Returns the voice_id (embedding_filename) from Chatterbox
    """
    try:
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Prepare request body for Chatterbox API
        request_body = {
            "input": {
                "action": "clone_voice",
                "audio_data": audio_base64,
                "voice_name": voice_name,
                "user_id": user_id,
            }
        }
        
        # Submit to Chatterbox API
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                CHATTERBOX_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {CHATTERBOX_API_KEY}",
                },
                json=request_body,
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Chatterbox API error: {response.text}"
                )
            
            result = response.json()
            
            # Handle async response (job ID)
            if "id" in result:
                job_id = result["id"]
                # Poll for completion (same logic as Flutter app)
                voice_id = await _poll_chatterbox_job(job_id)
                return {"voice_id": voice_id}
            
            # Handle direct response
            if result.get("status") == "success":
                embedding_filename = result.get("embedding_filename")
                if embedding_filename:
                    return {"voice_id": embedding_filename}
            
            # Handle error
            error_msg = result.get("error", "Unknown error from Chatterbox API")
            raise HTTPException(status_code=500, detail=f"Chatterbox error: {error_msg}")
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Chatterbox API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting to Chatterbox: {str(e)}")


async def _poll_chatterbox_job(job_id: str, max_attempts: int = 180) -> str:
    """Poll Chatterbox job status until completion"""
    status_url = f"{CHATTERBOX_API_URL.replace('/runsync', '').replace('/run', '')}/status/{job_id}"
    
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    status_url,
                    headers={"Authorization": f"Bearer {CHATTERBOX_API_KEY}"},
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    
                    if status == "COMPLETED":
                        output = data.get("output", {})
                        embedding_filename = output.get("embedding_filename")
                        if embedding_filename:
                            return embedding_filename
                        raise HTTPException(
                            status_code=500,
                            detail="Job completed but no voice_id returned"
                        )
                    elif status == "FAILED":
                        error = data.get("error", "Job failed")
                        raise HTTPException(status_code=500, detail=f"Job failed: {error}")
                    
                    # Still processing, wait and retry
                    import asyncio
                    await asyncio.sleep(1)
                else:
                    # Retry on error
                    import asyncio
                    await asyncio.sleep(1)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise HTTPException(
                    status_code=504,
                    detail=f"Job polling timeout: {str(e)}"
                )
            import asyncio
            await asyncio.sleep(1)
    
    raise HTTPException(status_code=504, detail="Job polling timeout (3 minutes)")


@app.get("/")
async def root():
    return {
        "service": "Web Voice Recording API",
        "status": "running",
        "firebase_available": FIREBASE_AVAILABLE,
        "chatterbox_url": CHATTERBOX_API_URL if CHATTERBOX_API_KEY else "Not configured",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "firebase_available": FIREBASE_AVAILABLE}


@app.post("/submit-voice")
async def submit_voice(
    token: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Submit a web-recorded voice via invitation token
    
    - **token**: Invitation token from the web recording link
    - **audio_file**: Audio file (WAV, MP3, etc.) - 3-10 seconds of clear speech
    """
    try:
        # Validate audio file type
        if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Audio file required (WAV, MP3, etc.)"
            )
        
        # Validate invitation token
        invitation = validate_invitation_token(token)
        if not invitation:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired invitation token"
            )
        
        # Get invitation data
        user_id = invitation.get("userId")  # Match Flutter VoiceInvitationService field name
        voice_name = invitation.get("voiceName")
        
        if not user_id or not voice_name:
            raise HTTPException(
                status_code=500,
                detail="Invalid invitation data: missing user_id or voice_name"
            )
        
        # Read audio file
        audio_bytes = await audio_file.read()
        
        if len(audio_bytes) < 1000:  # Less than 1KB is too small
            raise HTTPException(
                status_code=400,
                detail="Audio file too small. Please record at least 3 seconds of speech."
            )
        
        # Submit to Chatterbox API
        result = await submit_to_chatterbox(audio_bytes, user_id, voice_name)
        voice_id = result["voice_id"]
        
        # Update invitation in Firestore
        update_invitation(token, voice_id, status="completed")
        
        return {
            "success": True,
            "message": "Voice recorded and submitted successfully!",
            "voice_id": voice_id,
            "voice_name": voice_name,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error submitting voice: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error submitting voice: {str(e)}")


@app.get("/validate-token/{token}")
async def validate_token_endpoint(token: str):
    """
    Validate an invitation token (for web app to check before showing recording interface)
    Returns invitation details if valid, error if invalid
    """
    try:
        invitation = validate_invitation_token(token)
        if not invitation:
            raise HTTPException(
                status_code=404,
                detail="Invalid or expired invitation token"
            )
        
        return {
            "valid": True,
            "voice_name": invitation.get("voiceName"),
            "user_id": invitation.get("userId"),
            "created_at": invitation.get("createdAt"),
            "expires_at": invitation.get("expiresAt"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating token: {str(e)}")


# Alternative endpoints with /api/voice-invitation/ prefix for consistency
@app.get("/api/voice-invitation/validate")
async def validate_invitation_query(token: str):
    """
    Validate an invitation token (query parameter version)
    Compatible with: GET /api/voice-invitation/validate?token={token}
    """
    try:
        invitation = validate_invitation_token(token)
        if not invitation:
            raise HTTPException(
                status_code=404,
                detail="Invalid or expired invitation token"
            )
        
        return {
            "valid": True,
            "invitation": {
                "voice_name": invitation.get("voiceName"),
                "user_id": invitation.get("userId"),
                "recipient_name": invitation.get("recipientName"),
                "created_at": invitation.get("createdAt"),
                "expires_at": invitation.get("expiresAt"),
                "status": invitation.get("status", "pending"),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating token: {str(e)}")


@app.post("/api/voice-invitation/submit")
async def submit_invitation_voice(
    token: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Submit a web-recorded voice via invitation token (alternative endpoint)
    Compatible with: POST /api/voice-invitation/submit
    Same as /submit-voice but with different path for API consistency
    """
    return await submit_voice(token=token, audio_file=audio)


@app.get("/api/invitation/{token}")
async def get_invitation_status(token: str):
    """
    Get invitation status and details
    Returns invitation information including current status
    """
    if not FIREBASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured. Cannot retrieve invitation."
        )
    
    try:
        doc_ref = db.collection("voice_invitations").document(token)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=404,
                detail="Invitation not found"
            )
        
        data = doc.to_dict()
        if not data:
            raise HTTPException(
                status_code=404,
                detail="Invitation data not found"
            )
        
        return {
            "token": token,
            "status": data.get("status", "unknown"),
            "voice_name": data.get("voiceName"),
            "user_id": data.get("userId"),
            "recipient_name": data.get("recipientName"),
            "created_at": data.get("createdAt"),
            "expires_at": data.get("expiresAt"),
            "completed_at": data.get("completedAt"),
            "voice_id": data.get("voiceId"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving invitation: {str(e)}")


@app.get("/api/invitations/{user_id}")
async def get_user_invitations(user_id: str):
    """
    Get all invitations for a user
    Returns list of invitations (pending, completed, expired, cancelled)
    """
    if not FIREBASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured. Cannot retrieve invitations."
        )
    
    try:
        # Get all invitations for this user (limited to 100, sorted client-side)
        snapshot = db.collection("voice_invitations").where(
            "userId", "==", user_id
        ).limit(100).get()
        
        invitations = []
        for doc in snapshot:
            data = doc.to_dict()
            if data:
                invitations.append({
                    "token": doc.id,
                    "status": data.get("status", "unknown"),
                    "voice_name": data.get("voiceName"),
                    "recipient_name": data.get("recipientName"),
                    "created_at": data.get("createdAt"),
                    "expires_at": data.get("expiresAt"),
                    "completed_at": data.get("completedAt"),
                    "voice_id": data.get("voiceId"),
                })
        
        # Sort by created_at descending (newest first)
        invitations.sort(
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )
        
        return {
            "user_id": user_id,
            "count": len(invitations),
            "invitations": invitations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving invitations: {str(e)}")


@app.get("/api/invitations/{user_id}/completed")
async def get_completed_invitations(user_id: str):
    """
    Get only completed invitations for a user
    Useful for syncing new voices in Flutter app
    """
    if not FIREBASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Firebase is not configured. Cannot retrieve invitations."
        )
    
    try:
        # Get only completed invitations (limited to 100, sorted client-side)
        snapshot = db.collection("voice_invitations").where(
            "userId", "==", user_id
        ).where(
            "status", "==", "completed"
        ).limit(100).get()
        
        invitations = []
        for doc in snapshot:
            data = doc.to_dict()
            if data:
                invitations.append({
                    "token": doc.id,
                    "voice_name": data.get("voiceName"),
                    "recipient_name": data.get("recipientName"),
                    "voice_id": data.get("voiceId"),
                    "completed_at": data.get("completedAt"),
                    "created_at": data.get("createdAt"),
                })
        
        # Sort by completed_at descending (newest first)
        invitations.sort(
            key=lambda x: x.get("completed_at", ""), 
            reverse=True
        )
        
        return {
            "user_id": user_id,
            "count": len(invitations),
            "completed_invitations": invitations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving completed invitations: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Support PORT environment variable (for Railway, Render, Heroku, etc.)
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

