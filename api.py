from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os
import io
from gtts import gTTS
from main2 import AIProfessor, PROFESSOR_PROFILES, SlideContent  # Import necessary parts
from extract_info_from_upload import process_document # Import


# Create FastAPI app instance
app = FastAPI(
    title="AI Professor API",
    description="Backend API for the AI Professor application",
    version="0.1.0",
    docs_url="/docs",  # Enable interactive API documentation
    redoc_url="/redoc",  # Enable ReDoc documentation
)

# CORS Configuration
origins = [
    "http://localhost:5173",  # Vite default dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # Common React dev server
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

class InitialConversationResponse(BaseModel):
    object_id: str
    message: str
    audio_url: str
    num_pages: int

class ChatRequest(BaseModel):
    object_id: str
    message: str
    current_page: int

class ChatResponse(BaseModel):
    message: str
    current_page: int
    understanding_assessment: Dict[str, Any]
    audio_url: str
    end_of_conversation: bool = False # Flag

# --- In-Memory "Database" ---
file_data: Dict[str, Dict[str, Any]] = {}

# --- Helper Functions ---

async def get_ai_professor(professor_name: str) -> AIProfessor:
    """Dependency to get an AIProfessor instance."""
    return AIProfessor(professor_name)

def save_uploaded_file(file: UploadFile, object_id: str):
    """Saves the uploaded file to the objects directory."""
    _, ext = os.path.splitext(file.filename)  # Get extension
    file_path = os.path.join("objects", object_id + ext) #.pdf, .pptx
    os.makedirs("objects", exist_ok=True)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path


async def convert_text_to_speech_and_get_url(text: str) -> str:
     """
     Converts text to speech, saves audio, and returns URL.
     """
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
    """
    Uploads a file (PDF or PPTX), initializes the conversation.

    - **file**: The uploaded file (PDF or PPTX).
    - **start_page**: The page number to start on (default: 1).
    - **professor_name**: The name of the professor (default: "Andrew NG").

    Returns:
    - **object_id**:  Unique ID for the uploaded file.
    - **message**: Initial explanation from the professor.
    - **audio_url**: URL for the audio version of the initial message.
    - **num_pages**: Total number of pages in the document.
    """

    object_id = str(uuid.uuid4())
    file_path = save_uploaded_file(file, object_id)
    processed_content_path = os.path.join("processed_content", f"{object_id}.txt")
    os.makedirs("processed_content", exist_ok=True)

    try:
        # Use the external processing function!
        contents = process_document(file_path)
        
        # Save the processed contents
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
            "processed_content_path" : processed_content_path,
            "professor_name": professor_name,
            "conversation_history": ai_professor.conversation_history,
            "previous_explanations": ai_professor.previous_explanations,
            "num_pages" : num_pages
        }

        return InitialConversationResponse(
            object_id=object_id,
            message=explanation['prof_response']['explanation'],
            audio_url=audio_url,
            num_pages=num_pages
        )

    except Exception as e:
        print(f"Error during upload/initialization: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/chat", response_model=ChatResponse)
async def continue_chat(request: ChatRequest):
    """
    Continues the chat conversation.

    - **object_id**: The unique ID of the file.
    - **message**: The user's message.
    - **current_page**: The current page number.

    Returns:
      - **message**: The AI professor's response.
      - **current_page**: The updated current page number.
      - **understanding_assessment**:  Assessment of student understanding.
      - **audio_url**: URL for the audio version of the response.
      - **end_of_conversation**:  `True` if the conversation is over, else `False`
    """

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
                end_of_conversation=True
            )
        
        current_slide = next((slide for slide in slides if slide['page_number'] == ai_professor.current_page), None)
        if not current_slide: # Double check, though shouldn't happen.
             raise HTTPException(status_code=400, detail=f"Slide {ai_professor.current_page} not found.")
        
        response = await ai_professor.explain_slide(current_slide['content'], ai_professor.current_page)
        audio_url = await convert_text_to_speech_and_get_url(response['prof_response']['explanation'])


        file_data[request.object_id]["conversation_history"] = ai_professor.conversation_history
        file_data[request.object_id]["previous_explanations"] = ai_professor.previous_explanations

        return ChatResponse(
            message=response['prof_response']['explanation'],
            current_page=ai_professor.current_page,
            understanding_assessment=understanding,
            audio_url=audio_url
        )
    except Exception as e:
        print(f"Error during chat interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/audio/{audio_filename}")
async def get_audio(audio_filename: str):
    """Serves an audio file."""
    audio_filepath = os.path.join("audio", audio_filename)
    if not os.path.exists(audio_filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_filepath, media_type="audio/mpeg")