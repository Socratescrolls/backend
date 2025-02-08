import os
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import asyncio
def setup_environment():
    """Setup and validate environment variables"""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")
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

    async def explain_slide(self, slide_content: str, current_page: int, previous_response: Optional[str] = None) -> Dict[str, Any]:
        """Generate professor's explanation and actions for the current slide"""
        try:
            messages = [
                SystemMessage(content=f"""You are Professor {self.professor_name}. 
                Teaching Style: {self.profile['style']}
                Background: {self.profile['background']}
                Verification Style: {self.profile['verification_style']}
                
                You must respond with a JSON object containing:
                1. Your explanation of the slide content
                2. A question to verify understanding
                3. Decision to stay on current slide or move to next
                
                Previous response (if any): {previous_response}"""),
                
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
                    "prof_action": "stay/next",
                    "teaching_notes": {{
                        "difficulty_level": "basic/intermediate/advanced",
                        "prerequisites": ["prerequisite 1", "prerequisite 2"],
                        "suggested_exercises": ["exercise 1", "exercise 2"]
                    }}
                }}""")
            ]
            
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
            
        except Exception as e:
            print(f"Error generating professor response: {e}")
            raise

    async def continue_interaction(self, response: Dict[str, Any]) -> bool:
        """Ask user if they want to continue and get their next action"""
        while True:
            next_action = input("\nWhat would you like to do? (next/stay/end/help): ").strip().lower()
            
            if next_action == "help":
                print("\nAvailable commands:")
                print("- next: Move to the next slide")
                print("- stay: Stay on current slide for more explanation")
                print("- end: End the session")
                print("- help: Show this help message")
                continue
                
            if next_action in ["next", "stay", "end"]:
                if next_action == "next":
                    self.current_page += 1
                return next_action != "end"
            
            print("\nInvalid command. Type 'help' for available commands.")

    async def process_interaction(self, filename: str, current_page: int) -> None:
        try:
            # Read the file
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse slides
            slides = self.parse_slides(content)
            self.max_pages = len(slides)
            self.current_page = current_page
            
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
                print(f"\n=== Professor {self.professor_name}'s Response (Page {self.current_page}/{self.max_pages}) ===")
                print(f"\n{response['prof_response']['greeting']}" if response['prof_response'].get('greeting') else "")
                print(f"\nExplanation:\n{response['prof_response']['explanation']}")
                
                print("\nKey Points:")
                for point in response['prof_response']['key_points']:
                    print(f"- {point}")
                
                print(f"\nTo verify your understanding:\n{response['prof_response']['verification_question']}")
                
                # Get student's response
                answer = input("\nYour answer: ").strip()
                
                # Process student's response and get feedback
                follow_up = await self.explain_slide(current_slide['content'], self.current_page, answer)
                
                print(f"\nProfessor's feedback: {follow_up['prof_response']['explanation']}")
                
                # Ask user for next action
                continue_session = await self.continue_interaction(follow_up)
                
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            raise

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