import os
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import asyncio
import difflib

# Import the Teaching Assistant
from ai_teaching_assistant import AITeachingAssistant, run_quiz_interaction

def setup_environment():
    """Setup and validate environment variables"""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

PROFESSOR_PROFILES = {
   "Andrew NG": """You are teaching neural networks to a single student. Right, so imagine you're trying to detect cats in images - something I worked on at Google Brain. Um, let me break this down step by step.

First, we'll start with a simple "quick and dirty" implementation. Yeah, just a basic feedforward network. You know, it turns out that starting simple helps us understand what's really going on. Let's look at the math - don't worry if it seems complex at first.

See this activation function here? Okay, think of it like a filter that decides whether a neuron should fire or not. When I was teaching this at Stanford, students often got stuck here, but trust me, once you implement this yourself, it'll really be seared in your brain.

Let me show you a real example from my work with autonomous helicopters. There, we had to deal with similar activation patterns... Now, you need to go and code this yourself. Try starting with just one hidden layer - that's how I always recommend beginning.""",

   "David Malan": """You are teaching binary search to a single student. claps hands Alright! Let's dive into binary search! But before we get into the code, think about how you find a word in a dictionary. You don't go page by page, right? You jump to the middle and then decide whether to look left or right.

TL;DR: Binary search is like playing a number guessing game where you can only ask 'higher or lower' questions.

Here's some code - and don't worry if this looks intimidating at first, we'll break it down piece by piece. By the way, this algorithm is what powers the 'Find in Page' feature in your browser! Pretty cool, right?

typing on keyboard Let's see what happens when we run this... Boom! Did you see how fast that was? Now, there are some gotchas here - what happens if our list isn't sorted? That's where things get interesting...""",

   "John Guttag": """You are teaching sorting algorithms to a single student. Today we're discussing sorting algorithms, specifically merge sort. Let's analyze why this algorithm is significant for computational efficiency.

Consider an array of numbers: [3, 1, 4, 1, 5, 9]. We'll implement merge sort step by step:

First, examine the divide phase:
```python
def merge_sort(arr):
   if len(arr) <= 1:
       return arr
Notice the base case - this is crucial for preventing infinite recursion. Let's step through the execution, tracking the memory usage and time complexity at each stage.
Now, I want you to think about why we chose this approach. What would happen if we used a different base case?"""
}

class SlideContent(TypedDict):
    """Structure for slide content"""
    page_number: int
    content: str

class AIProfessor:
    def __init__(self, name: str):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI API
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o-mini",
            streaming=False
        )
        
        # Basic attributes
        self.name = name
        self.profile = PROFESSOR_PROFILES[name]
        self.current_page = 1
        self.max_pages = 1
        
        # Initialize conversation history and explanations
        self.conversation_history: List[Dict[str, Any]] = []
        self.previous_explanations: List[str] = []
        
        # Initialize teaching assistant
        try:
            self.teaching_assistant = AITeachingAssistant(name)
        except Exception as e:
            print(f"Error initializing teaching assistant: {e}")
            self.teaching_assistant = None
    
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

    async def ensure_teaching_assistant(self):
        """Ensure teaching assistant is initialized"""
        if self.teaching_assistant is None:
            self.teaching_assistant = AITeachingAssistant(self.name)
        return self.teaching_assistant

    async def evaluate_understanding(self, slide_content: str, student_response: str) -> Dict[str, Any]:
        """Evaluate student's understanding and decide next steps"""
        try:
            # Ensure teaching assistant is available
            teaching_assistant = await self.ensure_teaching_assistant()
            
            # Add student response to conversation history
            self.add_to_conversation_history("Student", student_response)
            
            messages = [
                SystemMessage(content=f"""{self.profile}
                
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
            context_message = f"""{self.profile}
            
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

    async def process_interaction(self, filename: str, current_page: int) -> None:
        try:
            # Read the file
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse slides
            slides = self.parse_slides(content)
            self.max_pages = len(slides)
            self.current_page = current_page
            
            # Reset conversation history for this session
            self.conversation_history = []
            self.previous_explanations = []
            
            # Add initial context to conversation history
            self.add_to_conversation_history("System", f"Starting lecture with Professor {self.name}")
            
            continue_session = True
            while continue_session:
                # Find current slide
                current_slide = next((slide for slide in slides 
                                    if slide['page_number'] == self.current_page), None)
                
                if not current_slide:
                    print(f"\nPage {self.current_page} not found in slides")
                    break
                
                # Get professor's explanation
                response = await self.explain_slide(current_slide['content'], self.current_page)
                
                # Print professor's response
                print(f"\n=== Professor {self.name}'s Response (Page {self.current_page}/{self.max_pages}) ===")
                print(f"\n{response['prof_response'].get('greeting', '')}")
                print(f"\nExplanation:\n{response['prof_response']['explanation']}")
                
                print("\nKey Points:")
                for point in response['prof_response']['key_points']:
                    print(f"- {point}")
                
                print(f"\nTo verify your understanding:\n{response['prof_response']['verification_question']}")
                
                # Get user's response
                student_response = input("\nYour answer: ").strip()
                
                # Evaluate student's understanding
                understanding = await self.evaluate_understanding(current_slide['content'], student_response)
                
                # Print professor's feedback
                print("\nProfessor's Feedback:")
                print(f"Understanding Level: {understanding['understanding_assessment']['level']}")
                print(f"Detailed Feedback: {understanding['understanding_assessment']['feedback']}")
                
                print("\nAreas to Improve:")
                for area in understanding['understanding_assessment']['areas_to_improve']:
                    print(f"- {area}")
                
                print(f"\nRecommended Action: {understanding['recommended_action']}")
                print(f"Reasoning: {understanding['reasoning']}")
                
                # Check if a quiz should be triggered
                quiz_result = await run_quiz_interaction(
                    self.teaching_assistant, 
                    self, 
                    current_slide
                )
                
                # Determine next action based on understanding assessment
                if understanding['recommended_action'] == 'next':
                    self.current_page += 1
                
                # Check if we've reached the last page
                continue_session = self.current_page <= self.max_pages
                
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            raise

    async def chat(self, message: str, current_page: int):
        if not self.teaching_assistant:
            raise ValueError("Teaching assistant not initialized")
            
        response = await self.teaching_assistant.chat(
            message=message,
            current_page=current_page
        )
        return response

async def main():
    try:
        print("\nWelcome to the AI Professor System!")
        print("\nAvailable Professors:")
        for name in PROFESSOR_PROFILES:
            print(f"- {name}")
        
        professor_name = input("\nPlease choose your professor: ").strip()
        if professor_name not in PROFESSOR_PROFILES:
            raise ValueError(f"Invalid professor name. Choose from: {', '.join(PROFESSOR_PROFILES.keys())}")
        
        professor = AIProfessor(professor_name)
        
        filename = input("Enter the filename containing slides: ").strip()
        current_page = int(input("Enter the page number to discuss: ").strip())
        
        await professor.process_interaction(filename, current_page)
        
        print("\nThank you for attending the session!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())