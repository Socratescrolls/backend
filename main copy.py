from typing import Dict, List, TypedDict, Optional, Any
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json
import asyncio
import re

def setup_environment():
    """Setup and validate environment variables"""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY2")

class SlideContent(TypedDict):
    """Structure for slide content"""
    page_number: int
    content: str

class CourseAdvisor:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o-mini",
            streaming=False
        )
        self.assessment_questions = [
            "What specific aspects of this paper would you like me to explain in more detail?",
            "Are you familiar with reinforcement learning concepts mentioned in the paper?",
            "Would you like me to explain any specific results or comparisons shown in the figures?",
            "Are you interested in the technical implementation details or more in the high-level concepts?"
        ]

    def parse_slides(self, content: str) -> List[SlideContent]:
        """Parse the slide content using the specific format"""
        # Split content into pages
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
        
        # Add the last page
        if current_page is not None:
            pages.append({
                'page_number': current_page,
                'content': '\n'.join(current_content).strip()
            })
            
        return pages

    async def analyze_paper(self, slides: List[SlideContent]) -> Dict[str, Any]:
        """Analyze the paper content"""
        try:
            # Create analysis prompt
            analysis_messages = [
                SystemMessage(content="""You are an expert at analyzing research papers. 
                Analyze this paper's content and provide a structured understanding."""),
                HumanMessage(content=f"""Analyze this research paper content:
                {json.dumps(slides, indent=2)}
                
                Provide analysis in this exact structure:
                {{
                    "title": "paper title",
                    "main_contributions": [
                        "contribution 1",
                        "contribution 2"
                    ],
                    "key_models": [
                        {{
                            "name": "model name",
                            "description": "model description",
                            "key_features": ["feature 1", "feature 2"]
                        }}
                    ],
                    "methodology": "brief description of methods",
                    "results_summary": "summary of main results",
                    "technical_concepts": [
                        {{
                            "concept": "concept name",
                            "explanation": "brief explanation"
                        }}
                    ]
                }}""")
            ]
            
            response = await self.llm.ainvoke(analysis_messages)
            return json.loads(response.content)
            
        except Exception as e:
            print(f"Error analyzing paper: {e}")
            raise

    async def process_paper(self, filename: str) -> Dict[str, Any]:
        try:
            # Read the file
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse slides
            slides = self.parse_slides(content)
            
            print("\n=== Paper Content Analysis ===")
            print(f"Found {len(slides)} pages")
            
            # Analyze paper
            analysis = await self.analyze_paper(slides)
            
            # Print initial analysis
            print(f"\nPaper Title: {analysis['title']}")
            print("\nMain Contributions:")
            for contribution in analysis['main_contributions']:
                print(f"- {contribution}")
            
            print("\nKey Models:")
            for model in analysis['key_models']:
                print(f"\n{model['name']}:")
                print(f"Description: {model['description']}")
                print("Key Features:")
                for feature in model['key_features']:
                    print(f"- {feature}")
            
            # Gather user questions
            print("\nTo provide better explanations, I'll ask a few questions:")
            responses = []
            for question in self.assessment_questions:
                print(f"\n{question}")
                response = input("Your answer: ").strip()
                responses.append(f"Q: {question}\nA: {response}")
            
            # Generate personalized explanation
            explanation_messages = [
                SystemMessage(content="You are an expert researcher providing detailed paper explanations."),
                HumanMessage(content=f"""Based on the paper content and user responses, provide a detailed explanation.
                
                Paper Analysis: {json.dumps(analysis, indent=2)}
                User Responses: {chr(10).join(responses)}
                
                Provide explanation in this structure:
                {{
                    "overview": "detailed overview targeted to user's interests",
                    "technical_explanations": [
                        {{
                            "topic": "topic name",
                            "explanation": "detailed explanation",
                            "relevance": "why this matters"
                        }}
                    ],
                    "practical_implications": ["implication 1", "implication 2"],
                    "suggested_readings": [
                        {{
                            "resource": "resource name",
                            "why_relevant": "explanation of relevance"
                        }}
                    ]
                }}""")
            ]
            
            explanation_response = await self.llm.ainvoke(explanation_messages)
            explanation = json.loads(explanation_response.content)
            
            # Print personalized explanation
            print("\n=== Personalized Explanation ===")
            print(f"\nOverview:\n{explanation['overview']}")
            
            print("\nDetailed Technical Explanations:")
            for topic in explanation['technical_explanations']:
                print(f"\n{topic['topic']}:")
                print(f"Explanation: {topic['explanation']}")
                print(f"Relevance: {topic['relevance']}")
            
            print("\nPractical Implications:")
            for implication in explanation['practical_implications']:
                print(f"- {implication}")
            
            print("\nSuggested Additional Reading:")
            for reading in explanation['suggested_readings']:
                print(f"\n- {reading['resource']}")
                print(f"  Why relevant: {reading['why_relevant']}")
            
            return {
                "slides": slides,
                "analysis": analysis,
                "explanation": explanation
            }
                
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            raise

async def main():
    try:
        advisor = CourseAdvisor()
        
        print("\nWelcome to the Research Paper Advisor!")
        print("Please provide the filename containing the paper content:")
        filename = input("Filename: ").strip()
        
        await advisor.process_paper(filename)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    setup_environment()
    asyncio.run(main())