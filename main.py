from typing import Dict, List, Tuple, Annotated, TypedDict, Optional
from dotenv import load_dotenv
import os
from langgraph.graph import Graph, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json
import asyncio
import sys
from pprint import pprint

# Load environment variables
def setup_environment():
    """Setup and validate environment variables"""
    # Load environment variables from .env file
    load_dotenv()
    # Set the correct environment variable name for OpenAI
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")

# Define course topics and difficulty levels
TOPICS = {
    "ML": ["Regression", "Classification", "Neural Networks", "Decision Trees"],
    "DL": ["CNN", "RNN", "Transformers", "GANs"],
    "NLP": ["Text Processing", "Word Embeddings", "Language Models", "Text Generation"],
}

DIFFICULTY_LEVELS = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3
}

class UniversityState(TypedDict):
    """State for the AI University workflow"""
    messages: List[Dict[str, str]]
    current_agent: str
    topic: str
    difficulty: int
    student_level: float
    quiz_scores: List[float]
    knowledge_assessment: Dict[str, float]
    final_score: Optional[float]
    needs_difficulty_adjustment: bool
    completed_topics: List[str]

class CourseAdvisor:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self._create_chain()

    def _create_chain(self):
        advisor_prompt = PromptTemplate(
            template="""You are an AI Course Advisor. Analyze the student's query and current knowledge to:
1. Identify the topic they want to learn
2. Assess their current knowledge level
3. Recommend appropriate difficulty level

Student Query: {query}

Respond in JSON format with this exact structure (do not add any whitespace or newlines):
{{"topic": "selected_topic", "initial_level": "beginner/intermediate/advanced", "knowledge_assessment": {{"current_understanding": 0.0, "key_gaps": ["gap1", "gap2"]}}}}
""",
            input_variables=["query"]
        )
        return advisor_prompt | self.llm | (lambda x: json.loads(x.content))

    async def process(self, state: UniversityState, query: str) -> UniversityState:
        assessment = await self.chain.ainvoke({"query": query})
        
        state["topic"] = assessment["topic"]
        state["difficulty"] = DIFFICULTY_LEVELS[assessment["initial_level"]]
        state["student_level"] = assessment["knowledge_assessment"]["current_understanding"]
        
        print("\nAdvisor Assessment:")
        print(f"Topic: {state['topic']}")
        print(f"Initial Level: {assessment['initial_level']}")
        print(f"Current Understanding: {state['student_level']:.2%}")
        
        state["current_agent"] = "professor"
        return state

class AIProfessor:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self._create_chain()

    def _create_chain(self):
        professor_prompt = PromptTemplate(
            template="""You are an AI Professor. Teach the student about {topic} at difficulty level {difficulty}/3.
Current student understanding: {student_level}
Previous messages: {messages}

Provide a focused lesson and determine if it's time for a quiz.

Respond in JSON format with this exact structure (do not add whitespace or newlines):
{{"lesson": "detailed explanation", "needs_quiz": true, "key_points": ["point1", "point2"], "next_step": "continue"}}
""",
            input_variables=["topic", "difficulty", "student_level", "messages"]
        )
        return professor_prompt | self.llm | (lambda x: json.loads(x.content))

    async def process(self, state: UniversityState) -> UniversityState:
        lesson_data = await self.chain.ainvoke({
            "topic": state["topic"],
            "difficulty": state["difficulty"],
            "student_level": state["student_level"],
            "messages": state["messages"][-3:] if state["messages"] else []
        })
        state["messages"].append({
            "role": "professor",
            "content": lesson_data["lesson"]
        })
        
        print("\nProfessor's Lesson:")
        print(lesson_data["lesson"])
        print("\nKey Points:")
        for point in lesson_data["key_points"]:
            print(f"- {point}")
        
        if lesson_data["needs_quiz"]:
            print("\nTime for a quiz to check understanding!")
            state["current_agent"] = "ta"
        else:
            state["current_agent"] = "professor"
            
        return state

class TeachingAssistant:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self._create_chain()

    def _create_chain(self):
        ta_prompt = PromptTemplate(
            template="""You are an AI Teaching Assistant. Create a quiz based on:
Topic: {topic}
Difficulty: {difficulty}/3
Key points covered: {messages}

Respond in JSON format with this exact structure (do not add whitespace or newlines):
{{"questions": [{{"question": "question text", "correct_answer": "answer", "explanation": "explanation"}}], "difficulty_adjustment": "maintain"}}
""",
            input_variables=["topic", "difficulty", "messages"]
        )
        return ta_prompt | self.llm | (lambda x: json.loads(x.content))

    async def process(self, state: UniversityState) -> UniversityState:
        quiz_data = await self.chain.ainvoke({
            "topic": state["topic"],
            "difficulty": state["difficulty"],
            "messages": state["messages"][-3:] if state["messages"] else []
        })
        
        print("\n=== Quiz Time! ===")
        score = 0
        total_questions = len(quiz_data["questions"])
        
        for i, q in enumerate(quiz_data["questions"], 1):
            print(f"\nQuestion {i}: {q['question']}")
            answer = input("Your answer: ").strip()
            
            if answer.lower() in q['correct_answer'].lower():
                score += 1
                print("Correct!")
                print(f"Explanation: {q['explanation']}")
            else:
                print(f"Incorrect. The correct answer was: {q['correct_answer']}")
                print(f"Explanation: {q['explanation']}")

        quiz_score = score / total_questions
        state["quiz_scores"].append(quiz_score)
        
        print(f"\nQuiz Score: {quiz_score:.2%}")
        
        if quiz_score < 0.6 and state["difficulty"] > 1:
            state["difficulty"] -= 1
            state["needs_difficulty_adjustment"] = True
            print("\nAdjusting difficulty level down for better learning.")
        elif quiz_score > 0.8 and state["difficulty"] < 3:
            state["difficulty"] += 1
            state["needs_difficulty_adjustment"] = True
            print("\nIncreasing difficulty level based on your performance.")
        
        state["current_agent"] = "professor" if quiz_score >= 0.7 else "ta"
        return state

class CourseAuditor:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self._create_chain()

    def _create_chain(self):
        auditor_prompt = PromptTemplate(
            template="""You are an AI Course Auditor. Review the student's performance:
Quiz scores: {quiz_scores}
Topics completed: {completed_topics}
Current level: {student_level}

Respond in JSON format with this exact structure (do not add whitespace or newlines):
{{"final_score": 85, "strengths": ["strength1", "strength2"], "areas_for_improvement": ["area1", "area2"], "certification_level": "intermediate", "detailed_feedback": "specific feedback about performance"}}
""",
            input_variables=["quiz_scores", "completed_topics", "student_level"]
        )
        return auditor_prompt | self.llm | (lambda x: json.loads(x.content))

    async def process(self, state: UniversityState) -> UniversityState:
        assessment = await self.chain.ainvoke({
            "quiz_scores": state["quiz_scores"],
            "completed_topics": state["completed_topics"],
            "student_level": state["student_level"]
        })
        state["final_score"] = assessment["final_score"]
        
        print("\n=== Final Assessment ===")
        print(f"Final Score: {assessment['final_score']}/100")
        print("\nStrengths:")
        for strength in assessment["strengths"]:
            print(f"- {strength}")
        print("\nAreas for Improvement:")
        for area in assessment["areas_for_improvement"]:
            print(f"- {area}")
        print(f"\nCertification Level: {assessment['certification_level'].title()}")
        print(f"\nDetailed Feedback:")
        print(assessment["detailed_feedback"])
        
        return state

class AIUniversity:
    def __init__(self):
        # Initialize LLM with specific parameters
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",  # Changed from gpt-4o-mini to gpt-4
            streaming=True
        )
        
        # Initialize agents
        self.advisor = CourseAdvisor(self.llm)
        self.professor = AIProfessor(self.llm)
        self.ta = TeachingAssistant(self.llm)
        self.auditor = CourseAuditor(self.llm)
        
        # Create workflow
        self.workflow = self._create_workflow()

    def should_proceed_to_quiz(self, state: UniversityState) -> bool:
        """Determine if we should move to quiz based on state"""
        return state["current_agent"] == "ta"

    def should_end_session(self, state: UniversityState) -> bool:
        """Determine if we should end the session"""
        return state["current_agent"] == "auditor"

    def _create_workflow(self):
        workflow = StateGraph(UniversityState)
        
        # Add nodes
        workflow.add_node("advisor", self.advisor.process)
        workflow.add_node("professor", self.professor.process)
        workflow.add_node("ta", self.ta.process)
        workflow.add_node("auditor", self.auditor.process)
        
        # Add edge from START to the entry point (advisor)
        workflow.add_edge(START, "advisor")
        
        # Add other edges between nodes
        workflow.add_edge("advisor", "professor")
        
        # Add conditional edges from professor
        workflow.add_conditional_edges(
            "professor",
            self.should_proceed_to_quiz,
            {
                True: "ta",
                False: "auditor"
            }
        )
        
        # Add edge from TA back to professor
        workflow.add_edge("ta", "professor")
        
        # Add edge from auditor to END
        workflow.add_edge("auditor", END)
        
        return workflow.compile()

async def run_interactive_session():
    try:
        university = AIUniversity()
        
        print("\nWelcome to AI University!")
        print("What would you like to learn about? (Please also mention your current knowledge level)")
        query = input("You: ")
        
        # Initialize state
        state = UniversityState(
            messages=[],
            current_agent="advisor",
            topic="",
            difficulty=1,
            student_level=0.0,
            quiz_scores=[],
            knowledge_assessment={},
            final_score=None,
            needs_difficulty_adjustment=False,
            completed_topics=[]
        )
        
        # Start with advisor
        state = await university.advisor.process(state, query)
        print(f"\nStarting topic: {state['topic']} at difficulty level {state['difficulty']}/3")
        
        # Main interaction loop
        while True:
            if state["current_agent"] == "professor":
                state = await university.professor.process(state)
                
                # Check if user wants to continue
                if len(state["messages"]) > 2:
                    cont = input("\nWould you like to continue learning? (yes/no): ").lower()
                    if cont != "yes":
                        state["current_agent"] = "auditor"
                        
            elif state["current_agent"] == "ta":
                state = await university.ta.process(state)
                
            elif state["current_agent"] == "auditor":
                state = await university.auditor.process(state)
                break
        
        return state
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    setup_environment()
    asyncio.run(run_interactive_session())