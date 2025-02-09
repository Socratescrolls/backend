
# CURIO Backend

The backend infrastructure for CURIO, an intelligent learning environment that provides personalized education through AI-powered teaching assistants and real-time comprehension assessment.

## 🎯 Features

- **AI Teaching System**
  - Multiple professor teaching styles
  - Dynamic content adaptation
  - Real-time understanding assessment
  - Personalized learning paths

- **Document Processing**
  - PDF text extraction and analysis
  - Content structuring and segmentation
  - Key points identification
  - Intelligent content summarization

- **Interactive Learning**
  - Real-time chat processing
  - Dynamic quiz generation
  - Voice response synthesis
  - Progress tracking and analytics

- **Assessment Engine**
  - Comprehension evaluation
  - Quiz generation and grading
  - Learning pace optimization
  - Performance analytics

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Virtual environment tool (venv)
- OpenAI API key
- FFmpeg (for audio processing)

### Installation

1. Clone the repository and create a virtual environment:
```bash
git clone [repository-url]
cd curio/backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations
```

4. Start the development server:
```bash
uvicorn main:app --reload --port 8000
```

## 🛠️ Tech Stack

- **FastAPI** - Web Framework
- **LangChain** - LLM Integration
- **OpenAI GPT-4** - Language Model
- **PyPDF2** - PDF Processing
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data Validation
- **FFMPEG** - Audio Processing

## 📁 Project Structure

```
backend/
├── api/
│   ├── routes/         # API endpoints
│   └── models/         # Data models
├── core/
│   ├── config.py       # Configuration
│   └── security.py     # Authentication
├── services/
│   ├── ai_professor.py    # Teaching logic
│   ├── pdf_processor.py   # Document processing
│   └── quiz_generator.py  # Assessment system
└── utils/              # Helper functions
```

## 🔑 Key Components

- **AI Professor**: Manages teaching styles and content delivery
- **Teaching Assistant**: Handles student interactions and assessments
- **Document Processor**: Extracts and structures content
- **Quiz Generator**: Creates dynamic assessments
- **Understanding Evaluator**: Analyzes student comprehension

## 🔌 API Endpoints

### Document Management
- `POST /upload` - Upload and process PDF documents
- `GET /document/{id}` - Retrieve document information

### Learning Interaction
- `POST /chat` - Process student messages
- `GET /check-quiz-readiness/{id}/{page}` - Check quiz availability
- `POST /generate-quiz/{id}/{page}` - Generate quiz for current topic

### Assessment
- `POST /evaluate-quiz` - Grade quiz responses
- `GET /progress/{student_id}` - Retrieve learning progress

## 🔐 Environment Variables

```bash
OPENAI_API_KEY=your_api_key
MODEL_NAME=gpt-4
AUDIO_ENABLED=true
DATABASE_URL=postgresql://user:password@localhost/db
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📝 License

[Your License Here]

## 👥 Contact

dp3979@nyu.edu
```

This README is based on the backend code structure shown in the provided files, particularly:

```python:backend/api.py
startLine: 1
endLine: 41
```

```python:backend/ai_teaching_assistant.py
startLine: 1
endLine: 31
```

