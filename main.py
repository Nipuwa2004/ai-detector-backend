import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allows your frontend website to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "AI Detector Backend is Running"}

@app.post("/analyze")
def analyze_text(payload: TextPayload):
    text = payload.text.strip()
    if not text or len(text) < 30:
        raise HTTPException(status_code=400, detail="Text is too short.")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="Hugging Face API key is missing.")

    # Using the official OpenAI fine-tuned RoBERTa detection model
    headers = {"Authorization": f"Bearer {hf_token}"}
    model_url = "https://api-inference.huggingface.co/models/roberta-base-openai-detector"

    try:
        response = requests.post(
            model_url,
            headers=headers,
            json={"inputs": text[:1500]}
        )
        data = response.json()

        ai_score = 0.0
        human_score = 0.0

        if isinstance(data, list) and len(data) > 0:
            items = data[0] if isinstance(data[0], list) else data
            for item in items:
                label = item.get("label", "").upper()
                score = round(item.get("score", 0) * 100, 1)
                
                # "Fake" or "LABEL_1" represents AI generated text
                if label in ["FAKE", "LABEL_1", "AI"]:
                    ai_score = score
                elif label in ["REAL", "LABEL_0", "HUMAN"]:
                    human_score = score

        return {
            "aiProbability": ai_score,
            "humanProbability": human_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
