import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import random

class AITeachingAssistant:
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
        
        # Tracking understanding and quiz readiness
        self.concept_understanding: Dict[str, float] = {}
        self.quiz_threshold = 0.7  # 70% understanding triggers quiz
    
    async def assess_concept_understanding(self, 
                                           conversation_history: List[Dict[str, Any]], 
                                           current_slide_content: str) -> Dict[str, Any]:
        """
        Assess the student's understanding of the current concept
        
        Args:
            conversation_history (List[Dict]): Full conversation history
            current_slide_content (str): Content of the current slide
        
        Returns:
            Dict containing understanding assessment
        """
        try:
            messages = [
                SystemMessage(content=f"""You are an AI Teaching Assistant monitoring student understanding.
                
                Carefully analyze the conversation history and current slide content to:
                1. Identify the key concepts being discussed
                2. Assess the student's level of understanding
                3. Determine if a quiz should be triggered
                
                Provide a detailed assessment in JSON format."""),
                
                HumanMessage(content=f"""Conversation History:
                {json.dumps(conversation_history, indent=2)}
                
                Current Slide Content:
                {current_slide_content}
                
                Respond with this exact structure:
                {{
                    "key_concepts": ["concept1", "concept2"],
                    "understanding_levels": {{
                        "concept1": "low/medium/high",
                        "concept2": "low/medium/high"
                    }},
                    "quiz_recommendation": {{
                        "trigger_quiz": true/false,
                        "reasoning": "explanation of quiz recommendation"
                    }}
                }}""")
            ]
            
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
        
        except Exception as e:
            print(f"Error assessing concept understanding: {e}")
            return {
                "key_concepts": [],
                "understanding_levels": {},
                "quiz_recommendation": {
                    "trigger_quiz": False,
                    "reasoning": "Error in assessment"
                }
            }
    
    async def generate_mcq_quiz(self, slide_content: str, key_concepts: List[str]) -> Dict[str, Any]:
        """
        Generate a Multiple Choice Questionnaire based on the slide content
        
        Args:
            slide_content (str): Content of the current slide
            key_concepts (List[str]): Key concepts to be tested
        
        Returns:
            Dict containing the MCQ quiz
        """
        try:
            messages = [
                SystemMessage(content=f"""You are an AI Teaching Assistant creating a Multiple Choice Quiz.
                
                Generate a quiz that:
                1. Covers the key concepts in the slide content
                2. Has 5 multiple-choice questions
                3. Includes varied difficulty levels
                4. Provides correct answers and explanations
                
                Ensure the quiz is educational and helps reinforce learning."""),
                
                HumanMessage(content=f"""Slide Content:
                {slide_content}
                
                Key Concepts to Test:
                {', '.join(key_concepts)}
                
                Respond with this exact structure:
                {{
                    "quiz_title": "Quiz Title",
                    "questions": [
                        {{
                            "id": "q1",
                            "question": "Question text",
                            "options": [
                                {{"id": "a", "text": "Option A"}},
                                {{"id": "b", "text": "Option B"}},
                                {{"id": "c", "text": "Option C"}},
                                {{"id": "d", "text": "Option D"}}
                            ],
                            "correct_answer": "a/b/c/d",
                            "explanation": "Detailed explanation of the correct answer"
                        }}
                        // 4 more questions following the same structure
                    ]
                }}""")
            ]
            
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
        
        except Exception as e:
            print(f"Error generating MCQ quiz: {e}")
            return {
                "quiz_title": "Concept Quiz",
                "questions": []
            }
    
    async def evaluate_quiz_performance(self, quiz: Dict[str, Any], student_answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate the student's quiz performance
        
        Args:
            quiz (Dict): The generated quiz
            student_answers (Dict): Student's submitted answers
        
        Returns:
            Dict containing quiz performance analysis
        """
        try:
            # Calculate score
            total_questions = len(quiz['questions'])
            correct_answers = 0
            detailed_results = []
            
            for question in quiz['questions']:
                student_answer = student_answers.get(question['id'])
                is_correct = student_answer == question['correct_answer']
                
                if is_correct:
                    correct_answers += 1
                
                detailed_results.append({
                    "question_id": question['id'],
                    "student_answer": student_answer,
                    "correct_answer": question['correct_answer'],
                    "is_correct": is_correct,
                    "explanation": question['explanation']
                })
            
            # Calculate performance metrics
            score_percentage = (correct_answers / total_questions) * 100
            performance_level = (
                "Excellent" if score_percentage >= 90 else
                "Good" if score_percentage >= 75 else
                "Satisfactory" if score_percentage >= 60 else
                "Needs Improvement"
            )
            
            return {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "score_percentage": score_percentage,
                "performance_level": performance_level,
                "detailed_results": detailed_results,
                "recommendation_for_professor": self._generate_teaching_recommendation(performance_level)
            }
        
        except Exception as e:
            print(f"Error evaluating quiz performance: {e}")
            return {
                "total_questions": 0,
                "correct_answers": 0,
                "score_percentage": 0,
                "performance_level": "Error",
                "detailed_results": [],
                "recommendation_for_professor": "Unable to evaluate quiz performance"
            }
    
    def _generate_teaching_recommendation(self, performance_level: str) -> str:
        """
        Generate teaching style recommendations based on student performance
        
        Args:
            performance_level (str): Student's performance level
        
        Returns:
            str: Recommendation for professor to adjust teaching style
        """
        recommendations = {
            "Excellent": "Student demonstrates high comprehension. Recommend introducing more advanced concepts and challenging examples.",
            "Good": "Student shows solid understanding. Suggest providing more practical applications and real-world scenarios.",
            "Satisfactory": "Student grasps basic concepts but needs more support. Recommend breaking down complex ideas, using more analogies, and providing additional explanations.",
            "Needs Improvement": "Student struggles with fundamental concepts. Suggest returning to foundational material, using simpler explanations, and providing more step-by-step guidance."
        }
        
        return recommendations.get(performance_level, "Unable to generate specific recommendation.")

async def run_quiz_interaction(teaching_assistant, professor, current_slide):
    """
    Run the quiz interaction process
    
    Args:
        teaching_assistant (AITeachingAssistant): The teaching assistant
        professor (AIProfessor): The professor
        current_slide (Dict): Current slide being discussed
    
    Returns:
        Dict: Quiz performance and recommendations
    """
    try:
        # Assess concept understanding
        understanding_assessment = await teaching_assistant.assess_concept_understanding(
            professor.conversation_history, 
            current_slide['content']
        )
        
        # Check if quiz should be triggered based on understanding level
        is_high_understanding = any(
            level == "high" for level in understanding_assessment['understanding_levels'].values()
        )
        
        # Check if quiz recommendation is triggered and understanding is high
        if (understanding_assessment['quiz_recommendation']['trigger_quiz'] and is_high_understanding):
            print("\n--- Quiz Time! ---")
            
            # Generate MCQ Quiz
            quiz = await teaching_assistant.generate_mcq_quiz(
                current_slide['content'], 
                understanding_assessment['key_concepts']
            )
            
            # Present Quiz to Student
            print(f"\nQuiz: {quiz['quiz_title']}")
            student_answers = {}
            
            for question in quiz['questions']:
                print(f"\n{question['question']}")
                for option in question['options']:
                    print(f"{option['id']}. {option['text']}")
                
                while True:
                    answer = input("\nYour answer (a/b/c/d): ").strip().lower()
                    if answer in ['a', 'b', 'c', 'd']:
                        student_answers[question['id']] = answer
                        break
                    print("Invalid input. Please enter a, b, c, or d.")
            
            # Evaluate Quiz Performance
            performance = await teaching_assistant.evaluate_quiz_performance(quiz, student_answers)
            
            # Print Quiz Results
            print("\n--- Quiz Results ---")
            print(f"Total Questions: {performance['total_questions']}")
            print(f"Correct Answers: {performance['correct_answers']}")
            print(f"Score: {performance['score_percentage']:.2f}%")
            print(f"Performance Level: {performance['performance_level']}")
            
            # Print Detailed Results
            print("\nDetailed Results:")
            for result in performance['detailed_results']:
                status = "✓" if result['is_correct'] else "✗"
                print(f"Q{result['question_id']}: {status} (Correct: {result['correct_answer']})")
                print(f"Explanation: {result['explanation']}\n")
            
            # Recommendation for Professor
            print("\nRecommendation for Professor:")
            print(performance['recommendation_for_professor'])
            
            return performance
        else:
            print("\nNot ready for quiz. Continue exploring the concept.")
            return None
    
    except Exception as e:
        print(f"Error during quiz interaction: {e}")
        return None