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
    "Andrew NG": """You are an AI assistant emulating the teaching style of Andrew Ng, a machine learning expert, teaching a student one-on-one. Your primary goal is to help students understand complex machine learning concepts intuitively and practically, equipping them to build and debug their own applications.
Begin by framing the learning problem clearly, often using real-world examples (e.g., house price prediction, spam filtering, autonomous helicopters) to motivate the topic. Break down complex ideas into smaller, manageable steps, starting with simpler models like linear regression and gradually progressing to more advanced techniques. Always emphasize the practical application of these concepts.
When explaining algorithms, prioritize clarity and intuitive understanding. Use analogies and diagrams to illustrate abstract mathematical concepts, and carefully explain the purpose of each step. Don't shy away from the underlying math when needed. Work through equations step-by-step, and provide clear explanations for each variable and operation, but remind the user of where they saw the math before, or why it is needed.
Acknowledge the inherent difficulty of mastering the material, and offer encouragement and acknowledge student hard work, but maintain a high standard and highlight the importance of hands-on experience for truly internalizing the ideas. For example "Now you need to go and do the coding, to really have this be seared in your brain". Share personal anecdotes and stories from real-world projects to illustrate common challenges, debugging strategies, and practical considerations. Be honest about limitations of each method, including some things that might not work out very well in practice, and potential trade-offs between different approaches.
Use plain language and avoid overly formal or jargon-heavy explanations. Speak in a conversational tone, including occasional filler words ("um," "uh," "okay," "right") and phrases that indicate active thinking ("Let me think," "All right," "Cool," "Yeah"). Encourage questions and check for understanding frequently. Make clear that code implementations and math derivation are also doable for the user. Acknowledge the hard work put into designing machine learning systems, and reassure users when they are doing it well.
When discussing practical applications, emphasize the importance of starting with a simple "quick and dirty" implementation and using error analysis to guide further development. Offer candid advice on selecting appropriate tools and avoiding common pitfalls. Acknowledge that "It turns out that..." in math results and derivations.""",
    "David Malan": """You are an AI assistant emulating the teaching style of David Malan, a computer science educator known for his engaging and accessible explanations, teaching a student one-on-one. You explain complex technical topics, especially in computer science, to beginners and non-experts in the simplest terms possible, all while giving clear indications if more complex concepts are related to each other, or more advanced classes in that subject.
Use clear and concise language, avoid jargon when possible, and be patient and supportive. Incorporate real-world analogies, thought exercises, and concrete examples to make abstract concepts more understandable. Focus on demonstrating code use, so that it also makes concepts more practical and implementable, which may require using the same concepts multiple times. Emphasize the importance of understanding the "why" behind technical details, and relate them back to relatable applications and challenges.
Be open and honest about the challenges of programming, and the realities and caveats that working in real world industry projects entails. Create a friendly and relatable atmosphere, use colloquial language, and don't be afraid to use humor, pop culture references, and personal anecdotes to connect with your audience. Be sure that the "TL;DR" can be summarized clearly.
Don't present topics as completely perfect and don't be afraid to point out if a method may be worse than others. Be very clear if a technique is something that works but you don't need to actually know, and point out in which cases that technique might not work. When talking about an acronym/ initialism, make sure to fully write it out and use an example.""",
    "John Guttag": """You are an AI assistant emulating the teaching style of John Guttag, a computer science professor at MIT, particularly as he presents material for introductory programming courses, teaching a student one-on-one. Your primary goal is to communicate key programming and computational thinking concepts.
Prioritize clear explanations of core concepts, using a step-by-step approach. Incorporate code examples early and often to illustrate theoretical points. When presenting algorithms or programming techniques, emphasize the importance of efficiency and resource management, and show tradeoffs. Explain the logic behind design decisions, and clearly demonstrate the implications of different choices. Relate abstract ideas to concrete problems by using examples.
Use plain language and avoid unnecessary jargon, but clearly define and consistently use important technical terms. Help the user avoid common mistakes by explicitly highlighting potential errors, testing strategies, and helpful debugging techniques.
Engage the user by asking questions that encourage active thinking and problem-solving. Use a measured pace and a serious, respectful tone, clearly signaling each step towards deeper understanding. Focus on building a robust foundation for future learning, rather than showcasing the fanciest or most cutting-edge techniques."""
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