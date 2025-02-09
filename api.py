from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
import io
from gtts import gTTS
from main import AIProfessor, PROFESSOR_PROFILES, SlideContent
from extract_info_from_upload import process_document
from ai_teaching_assistant import AITeachingAssistant

# Create FastAPI app instance
app = FastAPI(
    title="AI Professor API",
    description="Backend API for the AI Professor application",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class QuizQuestion(BaseModel):
    id: str
    question: str
    options: List[Dict[str, str]]

class Quiz(BaseModel):
    quiz_title: str
    questions: List[QuizQuestion]

class QuizAnswers(BaseModel):
    object_id: str
    current_page: int
    answers: Dict[str, str]

class QuizResult(BaseModel):
    score_percentage: float
    performance_level: str
    correct_answers: int
    total_questions: int
    detailed_results: List[Dict[str, Any]]
    recommendation_for_professor: str
    can_move_forward: bool

class InitialConversationResponse(BaseModel):
    object_id: str
    message: str
    audio_url: str
    num_pages: int
    verification_question: str
    key_points: List[str]

class ChatRequest(BaseModel):
    object_id: str
    message: str
    current_page: int

class ChatResponse(BaseModel):
    message: str
    current_page: int
    understanding_assessment: Dict[str, Any]
    audio_url: str
    end_of_conversation: bool = False
    verification_question: str = ""
    key_points: List[str] = []

# --- In-Memory "Database" ---
file_data: Dict[str, Dict[str, Any]] = {}

# --- Helper Functions ---
async def get_ai_professor(professor_name: str) -> AIProfessor:
    """Dependency to get an AIProfessor instance."""
    return AIProfessor(professor_name)

def save_uploaded_file(file: UploadFile, object_id: str):
    """Saves the uploaded file to the objects directory."""
    _, ext = os.path.splitext(file.filename)
    file_path = os.path.join("objects", object_id + ext)
    os.makedirs("objects", exist_ok=True)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path

async def convert_text_to_speech_and_get_url(text: str) -> str:
    """Converts text to speech, saves audio, and returns URL."""
    try:
        tts = gTTS(text)
        audio_stream = io.BytesIO()
        tts.save(audio_stream)
        audio_stream.seek(0)

        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_filepath = os.path.join("audio", audio_filename)
        os.makedirs("audio", exist_ok=True)

        with open(audio_filepath, "wb") as f:
            f.write(audio_stream.read())
        return f"/audio/{audio_filename}"
    except Exception as e:
        print(f"Text-to-speech conversion failed: {e}")
        raise HTTPException(status_code=500, detail="Text-to-speech conversion failed.")

# --- API Endpoints ---
@app.get("/professors", response_model=List[str])
async def get_professors():
    """Returns a list of available professor names."""
    return list(PROFESSOR_PROFILES.keys())

@app.post("/upload", response_model=InitialConversationResponse)
async def upload_file(
    file: UploadFile = File(...),
    start_page: int = Form(1),
    professor_name: str = Form("Andrew NG")
):
    """Uploads a file and initializes the conversation."""
    object_id = str(uuid.uuid4())
    file_path = save_uploaded_file(file, object_id)
    processed_content_path = os.path.join("processed_content", f"{object_id}.txt")
    os.makedirs("processed_content", exist_ok=True)

    try:
        contents = process_document(file_path)
        
        with open(processed_content_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(contents) if isinstance(contents, list) else contents)

        with open(processed_content_path, 'r', encoding='utf-8') as f:
            content = f.read()

        ai_professor = AIProfessor(professor_name)
        slides = ai_professor.parse_slides(content)
        num_pages = len(slides)

        if not 1 <= start_page <= num_pages:
            raise HTTPException(status_code=400, detail="Invalid start_page")

        ai_professor.current_page = start_page
        first_slide = next(slide for slide in slides if slide['page_number'] == start_page)
        explanation = await ai_professor.explain_slide(first_slide['content'], start_page)
        audio_url = await convert_text_to_speech_and_get_url(explanation['prof_response']['explanation'])

        file_data[object_id] = {
            "filename": file.filename,
            "filepath": file_path,
            "processed_content_path": processed_content_path,
            "professor_name": professor_name,
            "conversation_history": ai_professor.conversation_history,
            "previous_explanations": ai_professor.previous_explanations,
            "num_pages": num_pages,
            "quiz_results": []
        }

        return InitialConversationResponse(
            object_id=object_id,
            message=explanation['prof_response']['explanation'],
            audio_url=audio_url,
            num_pages=num_pages,
            verification_question=explanation['prof_response']['verification_question'],
            key_points=explanation['prof_response']['key_points']
        )

    except Exception as e:
        print(f"Error during upload/initialization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def continue_chat(request: ChatRequest):
    """Continues the chat conversation."""
    if request.object_id not in file_data:
        raise HTTPException(status_code=404, detail="File not found.")

    file_info = file_data[request.object_id]
    processed_content_path = file_info["processed_content_path"]

    try:
        with open(processed_content_path, 'r', encoding='utf-8') as f:
            content = f.read()

        ai_professor = AIProfessor(file_info["professor_name"])
        ai_professor.conversation_history = file_info["conversation_history"]
        ai_professor.previous_explanations = file_info["previous_explanations"]
        ai_professor.current_page = request.current_page

        # Ensure teaching assistant is initialized
        await ai_professor.ensure_teaching_assistant()

        slides = ai_professor.parse_slides(content)

        if not 1 <= request.current_page <= len(slides):
            raise HTTPException(status_code=400, detail="Invalid current_page")

        current_slide = next((slide for slide in slides if slide['page_number'] == request.current_page), None)
        if not current_slide:
            raise HTTPException(status_code=400, detail=f"Slide {request.current_page} not found.")

        understanding = await ai_professor.evaluate_understanding(current_slide['content'], request.message)

        if understanding['recommended_action'] == 'next':
            ai_professor.current_page += 1

        if ai_professor.current_page > len(slides):
            file_data[request.object_id]["conversation_history"] = ai_professor.conversation_history
            file_data[request.object_id]["previous_explanations"] = ai_professor.previous_explanations

            return ChatResponse(
                message="End of conversation.",
                current_page=request.current_page,
                understanding_assessment=understanding,
                audio_url="",
                end_of_conversation=True,
                verification_question="",
                key_points=[]
            )
        
        current_slide = next((slide for slide in slides if slide['page_number'] == ai_professor.current_page), None)
        if not current_slide:
            raise HTTPException(status_code=400, detail=f"Slide {ai_professor.current_page} not found.")
        
        response = await ai_professor.explain_slide(current_slide['content'], ai_professor.current_page)
        audio_url = await convert_text_to_speech_and_get_url(response['prof_response']['explanation'])

        file_data[request.object_id]["conversation_history"] = ai_professor.conversation_history
        file_data[request.object_id]["previous_explanations"] = ai_professor.previous_explanations

        return ChatResponse(
            message=response['prof_response']['explanation'],
            current_page=ai_professor.current_page,
            understanding_assessment=understanding,
            audio_url=audio_url,
            verification_question=response['prof_response']['verification_question'],
            key_points=response['prof_response']['key_points']
        )
    except Exception as e:
        print(f"Error during chat interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-quiz-readiness/{object_id}/{current_page}", response_model=Dict[str, Any])
async def check_quiz_readiness(object_id: str, current_page: int):
    """Check if a quiz should be triggered for the current page."""
    if object_id not in file_data:
        raise HTTPException(status_code=404, detail="File not found.")

    file_info = file_data[object_id]
    processed_content_path = file_info["processed_content_path"]

    try:
        with open(processed_content_path, 'r', encoding='utf-8') as f:
            content = f.read()

        ai_professor = AIProfessor(file_info["professor_name"])
        ai_professor.conversation_history = file_info["conversation_history"]
        slides = ai_professor.parse_slides(content)
        current_slide = next((slide for slide in slides if slide['page_number'] == current_page), None)

        if not current_slide:
            raise HTTPException(status_code=400, detail=f"Slide {current_page} not found.")

        understanding_assessment = await ai_professor.teaching_assistant.assess_concept_understanding(
            ai_professor.conversation_history,
            current_slide['content']
        )

        understanding_sufficient = any(
            level in ["high", "medium"] 
            for level in understanding_assessment['understanding_levels'].values()
        )

        return {
            "quiz_recommended": (understanding_assessment['quiz_recommendation']['trigger_quiz'] 
                               and understanding_sufficient),
            "reasoning": understanding_assessment['quiz_recommendation']['reasoning']
        }

    except Exception as e:
        print(f"Error checking quiz readiness: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-quiz/{object_id}/{current_page}", response_model=Quiz)
async def generate_quiz(object_id: str, current_page: int):
    """Generate a quiz for the current page if student is ready."""
    if object_id not in file_data:
        raise HTTPException(status_code=404, detail="File not found.")

    file_info = file_data[object_id]
    processed_content_path = file_info["processed_content_path"]

    try:
        with open(processed_content_path, 'r', encoding='utf-8') as f:
            content = f.read()

        ai_professor = AIProfessor(file_info["professor_name"])
        slides = ai_professor.parse_slides(content)
        current_slide = next((slide for slide in slides if slide['page_number'] == current_page), None)

        if not current_slide:
            raise HTTPException(status_code=400, detail=f"Slide {current_page} not found.")

        understanding = await ai_professor.teaching_assistant.assess_concept_understanding(
            ai_professor.conversation_history,
            current_slide['content']
        )

        quiz = await ai_professor.teaching_assistant.generate_mcq_quiz(
            current_slide['content'],
            understanding['key_concepts']
        )

        file_data[object_id]["current_quiz"] = quiz

        return quiz

    except Exception as e:
        print(f"Error generating quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-quiz", response_model=QuizResult)
async def evaluate_quiz(quiz_answers: QuizAnswers):
    """Evaluate the quiz answers and determine if student can move forward."""
    if quiz_answers.object_id not in file_data:
        raise HTTPException(status_code=404, detail="File not found.")

    file_info = file_data[quiz_answers.object_id]
    current_quiz = file_info.get("current_quiz")

    if not current_quiz:
        raise HTTPException(status_code=400, detail="No active quiz found.")

    try:
        ai_professor = AIProfessor(file_info["professor_name"])
        
        performance = await ai_professor.teaching_assistant.evaluate_quiz_performance(
            current_quiz,
            quiz_answers.answers
        )

        can_move_forward = performance['score_percentage'] >= 70

        file_data[quiz_answers.object_id]["quiz_results"].append(performance)

        return QuizResult(
            score_percentage=performance['score_percentage'],
            performance_level=performance['performance_level'],
            correct_answers=performance['correct_answers'],
            total_questions=performance['total_questions'],
            detailed_results=performance['detailed_results'],
            recommendation_for_professor=performance['recommendation_for_professor'],
            can_move_forward=can_move_forward
        )

    except Exception as e:
        print(f"Error evaluating quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{audio_filename}")
async def get_audio(audio_filename: str):
    """Serves an audio file."""
    audio_filepath = os.path.join("audio", audio_filename)
    if not os.path.exists(audio_filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_filepath, media_type="audio/mpeg")