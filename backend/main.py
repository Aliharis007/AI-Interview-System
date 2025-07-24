# backend/main.py
import os
import uuid
import shutil
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Any
# import speech_recognition as sr # Removed speech_recognition
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from faster_whisper import WhisperModel # Import Faster Whisper

# === Environment Setup ===
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"

aiplatform.init(project=PROJECT_ID, location=LOCATION)
gemini_model = GenerativeModel(MODEL_NAME) # Renamed to gemini_model to avoid conflict with whisper_model

app = FastAPI()

# Path to the JSON file containing job roles and questions
QUESTIONS_JSON_PATH = "questions.json"

# Global variable to store loaded questions data
all_job_roles_data: List[Dict] = []

# === Faster Whisper Model Initialization ===
# Choose a model size: 'tiny', 'base', 'small', 'medium', 'large-v2'
# Larger models are more accurate but require more resources (CPU/RAM/GPU) and take longer to download.
WHISPER_MODEL_SIZE = "base"
# Specify device: "cpu" or "cuda" (if you have an NVIDIA GPU)
# If you have a GPU, using "cuda" will be significantly faster.
WHISPER_DEVICE = "cpu"
# Set compute type for performance. "int8" is good for CPU, "float16" for GPU.
WHISPER_COMPUTE_TYPE = "int8"

# Initialize Whisper model globally on startup to avoid re-loading for each request
whisper_model = None

# === Evaluation Traits ===
EVALUATION_TRAITS = [
    "Resilience",
    "Self-Confidence",
    "Teamwork",
    "Influential",
    "Communication",
    "Ownership Mind-set",
    "Drive",
    "Discipline",
    "Creative Execution",
    "Customer Centricity"
]

# === Function to Load Questions from JSON ===
def load_questions_from_json():
    global all_job_roles_data
    try:
        with open(QUESTIONS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_job_roles_data = data.get("job_roles", [])
        print(f"Successfully loaded questions from {QUESTIONS_JSON_PATH}")
    except FileNotFoundError:
        print(f"Error: {QUESTIONS_JSON_PATH} not found. Please ensure it exists in the backend directory.")
        all_job_roles_data = []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {QUESTIONS_JSON_PATH}. Check file format.")
        all_job_roles_data = []
    except Exception as e:
        print(f"An unexpected error occurred while loading questions: {e}")
        all_job_roles_data = []

# Load questions and initialize Whisper model on startup
@app.on_event("startup")
def on_startup():
    load_questions_from_json()
    print("Questions data loaded on startup.")
    global whisper_model
    try:
        # Initialize Faster Whisper model. It will download the model weights if not present.
        print(f"Loading Faster Whisper model: {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} with {WHISPER_COMPUTE_TYPE} compute type...")
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        print("Faster Whisper model loaded successfully.")
    except Exception as e:
        print(f"Error loading Faster Whisper model: {e}")
        print("Transcription will not work. Please check faster-whisper installation and model download.")
        whisper_model = None # Set to None if loading fails


# === In-Memory Store for Interview Sessions ===
interview_sessions: Dict[str, Dict] = {}

# === Transcription Function (Now using Faster Whisper) ===
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using Faster Whisper.
    Args:
        file_path (str): The path to the audio file to transcribe.
    Returns:
        str: The transcribed text or an error message if transcription fails.
    """
    if whisper_model is None:
        return "(Transcription Failed: Whisper model not loaded)"

    try:
        # Transcribe the audio file.
        # language="ur" specifies Urdu. Without it, auto-detection happens.
        # beam_size can be adjusted for accuracy vs speed.
        segments, info = whisper_model.transcribe(file_path, language="ur", beam_size=5)
        
        transcription_text = ""
        for segment in segments:
            transcription_text += segment.text + " "
        
        return transcription_text.strip()
    except Exception as e:
        print(f"Faster Whisper Transcription Failed: {e}")
        return f"(Transcription Failed) {str(e)}"

# === Function to Evaluate Response with Gemini ===
async def evaluate_response_with_gemini(
    job_role_name: str,
    interview_data: List[Dict[str, Any]] # List of {"question_text": ..., "audio_transcription": ...}
) -> Dict[str, Any]:
    """
    Evaluates the candidate's interview responses using the Gemini model
    based on predefined traits and generates a concise report with scoring.
    """
    full_transcript = ""
    for i, qa in enumerate(interview_data):
        full_transcript += f"Question {i+1}: {qa['question_text']}\n"
        full_transcript += f"Candidate's Answer {i+1}: {qa['audio_transcription']}\n\n"

    prompt = f"""
    You are an AI interview evaluator for the role of "{job_role_name}".
    Below is the transcript of an interview. Your task is to evaluate the candidate's responses
    based on the following traits. Provide a score for each trait on a scale of 1 to 5,
    where 1 is Poor, 2 is Below Average, 3 is Average, 4 is Good, and 5 is Excellent.
    Also, provide a concise overall summary of the candidate's performance, highlighting
    strengths and areas for improvement, in a professional tone.

    Interview Transcript:
    {full_transcript}

    Evaluation Traits: {', '.join(EVALUATION_TRAITS)}

    Provide the output as a JSON object with the following structure:
    {{
        "scores": {{
            "Resilience": <score 1-5>,
            "Self-Confidence": <score 1-5>,
            "Teamwork": <score 1-5>,
            "Influential": <score 1-5>,
            "Communication": <score 1-5>,
            "Ownership Mind-set": <score 1-5>,
            "Drive": <score 1-5>,
            "Discipline": <score 1-5>,
            "Creative Execution": <score 1-5>,
            "Customer Centricity": <score 1-5>
        }},
        "overall_summary": "<concise summary of performance>"
    }}
    """

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "scores": {
                "type": "OBJECT",
                "properties": {trait: {"type": "INTEGER"} for trait in EVALUATION_TRAITS},
                "required": EVALUATION_TRAITS
            },
            "overall_summary": {"type": "STRING"}
        },
        "required": ["scores", "overall_summary"]
    }

    try:
        # Call Gemini model
        response = gemini_model.generate_content( # Use gemini_model here
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating evaluation report with Gemini: {e}")
        return {"error": f"Could not generate report: {e}"}


# === API Endpoint: Get All Job Roles ===
@app.get("/job_roles")
def get_job_roles():
    """
    Retrieves a list of all available job roles from the loaded JSON data.
    Returns:
        List[Dict]: A list of dictionaries, each representing a job role with its ID and name.
    """
    return [{"id": role["id"], "name": role["name"]} for role in all_job_roles_data]

# === API Endpoint: Start Interview ===
@app.post("/interview/start")
def start_interview(job_role_id: int = Form(...)):
    """
    Starts a new interview session for a specified job role.
    """
    job_role = next((role for role in all_job_roles_data if role["id"] == job_role_id), None)
    if not job_role:
        raise HTTPException(status_code=404, detail="Job role not found.")

    questions = job_role.get("questions", [])
    if not questions:
        raise HTTPException(status_code=404, detail="No questions found for this job role.")

    session_id = str(uuid.uuid4())
    interview_sessions[session_id] = {
        "job_role_id": job_role_id,
        "job_role_name": job_role["name"],
        "questions": questions,
        "answers": [],
        "transcripts": [],
        "current_q_index": 0,
        "evaluation_report": None
    }

    first_question = interview_sessions[session_id]["questions"][0]
    return {
        "session_id": session_id,
        "job_role_name": job_role["name"],
        "question": first_question["text"],
        "audio_path": first_question["audio_path"]
    }

# === API Endpoint: Submit Answer ===
@app.post("/interview/answer")
async def answer_question(session_id: str = Form(...), audio: UploadFile = File(...)):
    """
    Submits an audio answer for the current question in an interview session.
    """
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = interview_sessions[session_id]
    current_q_index = session["current_q_index"]
    questions = session["questions"]

    if current_q_index >= len(questions):
        return {"completed": True, "message": "Interview already completed."}

    current_question = questions[current_q_index]

    audio_path = f"temp_{session_id}_{current_q_index}.wav"
    transcription = "(Transcription Failed)"
    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        transcription = transcribe_audio(audio_path) # Call Faster Whisper transcription
    except Exception as e:
        print(f"Error processing audio or transcribing: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    session["answers"].append({
        "question_id": current_question["id"],
        "question_text": current_question["text"],
        "audio_transcription": transcription
    })
    session["transcripts"].append(transcription)
    session["current_q_index"] += 1

    if session["current_q_index"] < len(questions):
        next_question = questions[session["current_q_index"]]
        return {
            "completed": False,
            "transcription": transcription,
            "next_question": next_question["text"],
            "next_audio_path": next_question["audio_path"],
            "question_index": session["current_q_index"]
        }
    else:
        return {
            "completed": True,
            "transcription": transcription,
            "message": "Interview completed. You can now generate the report."
        }

# === API Endpoint: Get Interview Report ===
@app.get("/interview/get_report/{session_id}")
async def get_interview_report(session_id: str):
    """
    Generates and retrieves the evaluation report for a completed interview session.
    """
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = interview_sessions[session_id]

    if session["current_q_index"] < len(session["questions"]):
        raise HTTPException(status_code=400, detail="Interview not yet completed.")

    if session["evaluation_report"]:
        return session["evaluation_report"]

    job_role_name = session["job_role_name"]
    interview_answers = session["answers"]

    report = await evaluate_response_with_gemini(job_role_name, interview_answers)
    session["evaluation_report"] = report

    return report
