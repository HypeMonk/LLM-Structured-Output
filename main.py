from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import os
import requests
import json
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

AI_API_TOKEN = os.getenv("AI_API_TOKEN")
CHAT_URL = os.getenv("CHAT_URL")

if not AI_API_TOKEN:
    raise RuntimeError("AI_API_TOKEN not set")

if not CHAT_URL:
    raise RuntimeError("CHAT_URL not set")

app = FastAPI()

# -----------------------------
# Allow ALL origins (CORS)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Model
# -----------------------------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

# -----------------------------
# Response Model
# -----------------------------
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(request: CommentRequest):

    try:
        payload = {
            "model": "gpt-4.1-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a sentiment analysis API. Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze this comment and respond ONLY in this exact JSON format:

{{
  "sentiment": "positive | negative | neutral",
  "rating": 1-5
}}

Rules:
- 5 = highly positive
- 4 = positive
- 3 = neutral
- 2 = negative
- 1 = highly negative
- No explanations.
- No extra text.

Comment:
{request.comment}
"""
                }
            ],
            "temperature": 0
        }

        headers = {
            "Authorization": f"Bearer {AI_API_TOKEN}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            CHAT_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=response.text)

        data = response.json()

        # Extract response safely
        content = data["choices"][0]["message"]["content"]

        # Remove accidental markdown formatting
        content = content.strip().replace("```json", "").replace("```", "")

        parsed = json.loads(content)

        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model did not return valid JSON")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))