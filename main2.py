import os
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import asyncio
import difflib
from extract_info_from_upload import process_document # Import

def setup_environment():
    """Setup and validate environment variables"""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")  # Ensure correct key

PROFESSOR_PROFILES = {
    "Andrew NG": {
        "style": "Focuses on practical applications and real-world examples. Breaks down complex ML concepts into digestible pieces. Often uses analogies and visual explanations.",
        "background": "Expert in Machine Learning and AI. Known for making complex concepts accessible.",
        "verification_style": "Uses step-by-step verification of understanding, often asking students to explain concepts back."
    },
    "David Malan": {
        "style": "Energetic and engaging. Uses live demonstrations and interactive examples. Builds concepts from first principles.",
        "background": "Computer Science educator known for CS50. Expert at making technical concepts approachable.",
        "verification_style": "Uses 'show of hands' style questions and encourages active participation."
    },
    "John Guttag": {
        "style": "Methodical and thorough. Emphasizes theoretical foundations while connecting to practical applications. Uses mathematical reasoning.",
        "background": "Expert in Computer Science and Programming. Known for rigorous but clear explanations.",
        "verification_style": "Asks probing questions to verify deep understanding of concepts."
    }
}

class SlideContent(TypedDict):
    """Structure for slide content"""
    page_number: int
    content: str

class AIProfessor:
    def __init__(self, professor_name: str):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI API
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o-mini",
            streaming=False
        )
        self.professor_name = professor_name
        self.profile = PROFESSOR_PROFILES[professor_name]
        self.current_page = 1
        self.max_pages = 1
        
        # Track conversation history manually
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Track previous explanations to prevent repetition
        self.previous_explanations: List[str] = []
    
    def add_to_conversation_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history"""
        entry = {
            "role": role,
            "content": content,
            "page": self.current_page
        }
        if metadata:
            entry.update(metadata)
        self.conversation_history.append(entry)
    
    def get_conversation_context(self) -> str:
        """Retrieve the conversation context as a formatted string"""
        context = "Conversation History:\n"
        for message in self.conversation_history:
            context += f"Page {message.get('page', 'N/A')} - {message['role']}: {message['content']}\n"
        return context.strip()
    
    def check_explanation_similarity(self, new_explanation: str, threshold: float = 0.8) -> bool:
        """
        Check if the new explanation is too similar to previous explanations
        Returns True if the explanation is too similar, False otherwise
        """
        for prev_explanation in self.previous_explanations:
            # Use difflib to calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, prev_explanation, new_explanation).ratio()
            if similarity > threshold:
                return True
        return False
    
    def parse_slides(self, content: str) -> List[SlideContent]:
        """Parse the slide content using the specific format"""
        pages = []
        current_page = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('Page '):
                if current_page is not None:
                    pages.append({
                        'page_number': current_page,
                        'content': '\n'.join(current_content).strip()
                    })
                current_page = int(line.split(':')[0].split()[1])
                current_content = []
            elif line.strip() == 'Text content:':
                continue
            else:
                current_content.append(line)
        
        if current_page is not None:
            pages.append({
                'page_number': current_page,
                'content': '\n'.join(current_content).strip()
            })
            
        return pages

    async def evaluate_understanding(self, slide_content: str, student_response: str) -> Dict[str, Any]:
        """Evaluate student's understanding and decide next steps"""
        try:
            # Add student response to conversation history
            self.add_to_conversation_history("Student", student_response)
            
            messages = [
                SystemMessage(content=f"""You are Professor {self.professor_name}. 
                Teaching Style: {self.profile['style']}
                Background: {self.profile['background']}
                Verification Style: {self.profile['verification_style']}
                
                Evaluate the student's response to the slide content and provide:
                1. Feedback on their understanding
                2. Recommendation to stay or move to next slide
                3. Reasoning for your decision
                
                Previous Conversation Context:
                {self.get_conversation_context()}
                
                Respond with a JSON object containing these details."""),
                
                HumanMessage(content=f"""Slide Content:
                {slide_content}
                
                Student Response:
                {student_response}
                
                Respond with this exact structure:
                {{
                    "understanding_assessment": {{
                        "level": "low/medium/high",
                        "feedback": "detailed explanation of student's understanding",
                        "areas_to_improve": ["area 1", "area 2"]
                    }},
                    "recommended_action": "stay/next",
                    "reasoning": "explanation of why to stay or move"
                }}""")
            ]
            
            response = await self.llm.ainvoke(messages)
            understanding = json.loads(response.content)
            
            # Add professor's assessment to conversation history
            self.add_to_conversation_history("Professor", json.dumps(understanding))
            
            return understanding
            
        except Exception as e:
            print(f"Error evaluating student understanding: {e}")
            raise

    async def explain_slide(self, slide_content: str, current_page: int) -> Dict[str, Any]:
        """Generate professor's explanation for the current slide"""
        try:
            # Prepare context with anti-repetition guidance
            context_message = f"""You are Professor {self.professor_name}. 
            Teaching Style: {self.profile['style']}
            Background: {self.profile['background']}
            Verification Style: {self.profile['verification_style']}
            
            IMPORTANT: Avoid repeating previous explanations. 
            If your explanation is too similar to past explanations, provide a 
            substantially different approach, such as:
            - Using a completely different analogy
            - Focusing on different aspects of the topic
            - Changing the level of detail
            - Providing a contrasting perspective
            
            Previous Conversation Context:
            {self.get_conversation_context()}
            """
            
            messages = [
                SystemMessage(content=context_message),
                
                HumanMessage(content=f"""Current slide (Page {current_page}):
                {slide_content}
                
                Respond with this exact structure:
                {{
                    "prof_response": {{
                        "greeting": "optional greeting",
                        "explanation": "detailed explanation in your teaching style",
                        "key_points": ["point 1", "point 2"],
                        "verification_question": "question to check understanding"
                    }},
                    "teaching_notes": {{
                        "difficulty_level": "basic/intermediate/advanced",
                        "prerequisites": ["prerequisite 1", "prerequisite 2"],
                        "suggested_exercises": ["exercise 1", "exercise 2"]
                    }}
                }}""")
            ]
            
            response = await self.llm.ainvoke(messages)
            explanation = json.loads(response.content)
            
            # Check for explanation similarity and regenerate if too similar
            explanation_text = explanation['prof_response']['explanation']
            max_attempts = 3
            attempt = 0
            
            while (self.check_explanation_similarity(explanation_text) and attempt < max_attempts):
                # If too similar, regenerate with added guidance
                context_message += "\nPrevious explanation was too similar. Generate a COMPLETELY DIFFERENT explanation."
                messages[0] = SystemMessage(content=context_message)
                
                response = await self.llm.ainvoke(messages)
                explanation = json.loads(response.content)
                explanation_text = explanation['prof_response']['explanation']
                attempt += 1
            
            # Store the explanation to prevent future repetitions
            self.previous_explanations.append(explanation_text)
            
            # Add professor's explanation to conversation history
            self.add_to_conversation_history("Professor", explanation_text, 
                                             metadata={"explanation_type": "slide_explanation"})
            
            return explanation
            
        except Exception as e:
            print(f"Error generating professor response: {e}")
            raise