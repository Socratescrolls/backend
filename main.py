import os
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import asyncio
import difflib
import logging
from datetime import datetime

# Import the Teaching Assistant and Course Auditor
from ai_teaching_assistant import AITeachingAssistant, run_quiz_interaction
from ai_course_auditor import CourseAuditor, AuditorError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        # Track conversation history manually
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Track previous explanations to prevent repetition
        self.previous_explanations: List[str] = []
        
        # Initialize Teaching Assistant and Course Auditor
        self.teaching_assistant = AITeachingAssistant(professor_name)
        self.course_auditor = CourseAuditor()
        
        # Track quiz results
        self.quiz_results: List[Dict[str, Any]] = []
        
        # Track session metadata
        self.session_start_time = datetime.now()
        self.session_metadata = {
            "professor_name": professor_name,
            "profile": PROFESSOR_PROFILES[professor_name],
            "start_time": self.session_start_time.isoformat()
        }
    
    def add_to_conversation_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history"""
        entry = {
            "role": role,
            "content": content,
            "page": self.current_page,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            entry.update(metadata)
        self.conversation_history.append(entry)
    
    def get_conversation_context(self) -> str:
        """Retrieve the conversation context as a formatted string"""
        context = "Conversation History:\n"
        for message in self.conversation_history[-5:]:  # Last 5 messages for context
            context += f"Page {message.get('page', 'N/A')} - {message['role']}: {message['content']}\n"
        return context.strip()
    
    def check_explanation_similarity(self, new_explanation: str, threshold: float = 0.8) -> bool:
        """Check if the new explanation is too similar to previous explanations"""
        for prev_explanation in self.previous_explanations[-3:]:  # Check last 3 explanations
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
                {self.get_conversation_context()}"""),
                
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
            
            self.add_to_conversation_history(
                "Professor", 
                json.dumps(understanding),
                metadata={"assessment_type": "understanding_evaluation"}
            )
            
            return understanding
            
        except Exception as e:
            logger.error(f"Error evaluating understanding: {str(e)}")
            raise

    async def explain_slide(self, slide_content: str, current_page: int) -> Dict[str, Any]:
        """Generate professor's explanation for the current slide"""
        try:
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
            
            explanation_text = explanation['prof_response']['explanation']
            max_attempts = 3
            attempt = 0
            
            while (self.check_explanation_similarity(explanation_text) and attempt < max_attempts):
                context_message += "\nPrevious explanation was too similar. Generate a COMPLETELY DIFFERENT explanation."
                messages[0] = SystemMessage(content=context_message)
                
                response = await self.llm.ainvoke(messages)
                explanation = json.loads(response.content)
                explanation_text = explanation['prof_response']['explanation']
                attempt += 1
            
            self.previous_explanations.append(explanation_text)
            
            self.add_to_conversation_history(
                "Professor", 
                explanation_text,
                metadata={
                    "explanation_type": "slide_explanation",
                    "teaching_notes": explanation['teaching_notes']
                }
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise

    async def generate_session_report(self) -> None:
        """Generate and display the final session report"""
        try:
            logger.info("Generating final session report...")
            
            self.session_metadata["end_time"] = datetime.now().isoformat()
            self.session_metadata["duration"] = str(datetime.now() - self.session_start_time)
            
            report = await self.course_auditor.generate_final_report(
                self.conversation_history,
                self.quiz_results
            )
            
            self._display_audit_report(report)
            self._save_audit_report(report)
            
        except AuditorError as e:
            logger.error(f"Failed to generate audit report: {str(e)}")
            print("\nUnable to generate complete audit report due to an error.")
            print("Please contact support with the following error message:")
            print(str(e))
        except Exception as e:
            logger.error(f"Unexpected error in generate_session_report: {str(e)}")
            raise

    def _display_audit_report(self, report: Dict[str, Any]) -> None:
        """Display the audit report in a formatted way"""
        try:
            print("\n" + "="*50)
            print("COURSE SESSION AUDIT REPORT")
            print("="*50)
            
            print(f"\nSession Details:")
            print(f"Generated: {report['report_metadata']['generated_at']}")
            print(f"Total Score: {report['report_metadata']['total_score']}%")
            print(f"Performance Level: {report['report_metadata']['performance_level']}")
            
            print("\nPerformance Metrics:")
            for metric in report['visualization_data']['metrics']:
                print(f"- {metric['name']}: {metric['percentage']}%")
            
            print("\nLearning Profile:")
            print(f"Preferred Style: {report['learning_profile']['preferred_style']}")
            print("\nMost Effective Topics:")
            for topic in report['learning_profile']['effective_topics']:
                print(f"- {topic}")
            
            print("\nProgress Analysis:")
            print(f"Initial Level: {report['progress_analysis']['initial_level']}")
            print(f"Final Level: {report['progress_analysis']['final_level']}")
            
            print("\nKey Improvements:")
            for improvement in report['progress_analysis']['key_improvements']:
                print(f"- {improvement}")
            
            print("\nAreas Needing Focus:")
            for area in report['progress_analysis']['challenging_areas']:
                print(f"- {area}")
            
            print("\nRecommendations:")
            for section in ['key_strengths', 'improvement_areas', 'action_items', 'additional_resources']:
                print(f"\n{section.replace('_', ' ').title()}:")
                for item in report['recommendations'][section]:
                    print(f"- {item}")
            
        except Exception as e:
            logger.error(f"Error displaying audit report: {str(e)}")
            print("\nError displaying complete report. Basic metrics available in saved report file.")

    def _save_audit_report(self, report: Dict[str, Any]) -> None:
        """Save the audit report to a file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_report_{timestamp}.json"
            
            report['session_metadata'] = self.session_metadata
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Audit report saved to {filename}")
            print(f"\nFull report saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving audit report: {str(e)}")
            print("\nError saving audit report to file.")

    async def process_interaction(self, filename: str, current_page: int) -> None:
        """Process the entire teaching session"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
            
            slides = self.parse_slides(content)
            self.max_pages = len(slides)
            self.current_page = current_page
            
            self.conversation_history = []
            self.previous_explanations = []
            
            self.add_to_conversation_history(
                "System", 
                f"Starting lecture with Professor {self.professor_name}"
            )
            
            continue_session = True
            while continue_session:
                current_slide = next(
                    (slide for slide in slides if slide['page_number'] == self.current_page), 
                    None
                )
                
                if not current_slide:
                    print(f"\nPage {self.current_page} not found in slides")
                    break
                
                response = await self.explain_slide(current_slide['content'], self.current_page)
                
                print(f"\n=== Professor {self.professor_name}'s Response (Page {self.current_page}/{self.max_pages}) ===")
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
                
                # Store quiz result if one was generated
                if quiz_result:
                    self.quiz_results.append(quiz_result)
                
                # Determine next action based on understanding assessment
                if understanding['recommended_action'] == 'next':
                    self.current_page += 1
                
                # Check if we've reached the last page
                continue_session = self.current_page <= self.max_pages
            
            # After session ends, generate audit report
            await self.generate_session_report()
            
        except Exception as e:
            logger.error(f"Unexpected error in process_interaction: {str(e)}")
            raise

async def main():
    """Main entry point for the AI Professor System"""
    try:
        print("\nWelcome to the AI Professor System!")
        print("\nAvailable Professors:")
        for name in PROFESSOR_PROFILES:
            print(f"- {name}")
        
        while True:
            professor_name = input("\nPlease choose your professor: ").strip()
            if professor_name in PROFESSOR_PROFILES:
                break
            print(f"Invalid professor name. Choose from: {', '.join(PROFESSOR_PROFILES.keys())}")
        
        professor = AIProfessor(professor_name)
        
        while True:
            try:
                filename = input("Enter the filename containing slides: ").strip()
                with open(filename, 'r', encoding='utf-8') as f:
                    # Just try to read the file to verify it exists and is readable
                    f.read()
                break
            except FileNotFoundError:
                print(f"File '{filename}' not found. Please check the filename and try again.")
            except Exception as e:
                print(f"Error reading file: {str(e)}")
        
        while True:
            try:
                current_page = int(input("Enter the page number to discuss: ").strip())
                if current_page > 0:
                    break
                print("Page number must be positive.")
            except ValueError:
                print("Please enter a valid number.")
        
        await professor.process_interaction(filename, current_page)
        
        print("\nThank you for attending the session!")
        
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        print(f"An error occurred: {str(e)}")
    finally:
        print("\nSession ended. Thank you for using the AI Professor System!")

if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())