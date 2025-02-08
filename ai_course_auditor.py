from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import statistics
from datetime import datetime
import logging
from json.decoder import JSONDecodeError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AuditorError(Exception):
    """Base exception class for CourseAuditor"""
    pass

class AnalysisError(AuditorError):
    """Raised when analysis fails"""
    pass

class MetricsCalculationError(AuditorError):
    """Raised when metrics calculation fails"""
    pass

class CourseAuditor:
    def __init__(self):
        """Initialize the Course Auditor with necessary components"""
        try:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-4o-mini",
                streaming=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
            raise AuditorError(f"LLM initialization failed: {str(e)}")
        
        # Performance metrics weights
        self.weights = {
            "quiz_performance": 0.35,
            "engagement_quality": 0.25,
            "concept_understanding": 0.25,
            "progress_rate": 0.15
        }

    async def analyze_conversation(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the entire conversation history to evaluate student performance
        
        Args:
            conversation_history: List of conversation entries with roles, content, and metadata
            
        Returns:
            Dict containing comprehensive analysis and evaluation
            
        Raises:
            AnalysisError: If analysis fails
            JSONDecodeError: If response parsing fails
        """
        try:
            if not conversation_history:
                raise AnalysisError("Empty conversation history provided")

            messages = [
                SystemMessage(content="""You are an expert Course Auditor.
                Analyze the entire conversation history to:
                1. Evaluate student engagement and participation
                2. Assess concept understanding progression
                3. Identify strengths and areas for improvement
                4. Provide specific, actionable recommendations
                
                Focus on both quantitative and qualitative aspects."""),
                
                HumanMessage(content=f"""Conversation History:
                {json.dumps(conversation_history, indent=2)}
                
                Provide a detailed analysis following this structure:
                {{
                    "engagement_metrics": {{
                        "participation_rate": float,
                        "response_quality": float,
                        "question_asking_frequency": float
                    }},
                    "understanding_progression": {{
                        "initial_level": float,
                        "final_level": float,
                        "key_improvements": [str],
                        "challenging_areas": [str]
                    }},
                    "learning_patterns": {{
                        "preferred_learning_style": str,
                        "most_effective_topics": [str],
                        "attention_span": str
                    }}
                }}""")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Validate response structure
            analysis_data = json.loads(response.content)
            self._validate_analysis_data(analysis_data)
            
            return analysis_data
            
        except JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise AnalysisError(f"Failed to parse analysis response: {str(e)}")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise AnalysisError(f"Analysis failed: {str(e)}")

    def _validate_analysis_data(self, data: Dict[str, Any]) -> None:
        """Validate the structure of analysis data"""
        required_keys = {
            "engagement_metrics": ["participation_rate", "response_quality", "question_asking_frequency"],
            "understanding_progression": ["initial_level", "final_level", "key_improvements", "challenging_areas"],
            "learning_patterns": ["preferred_learning_style", "most_effective_topics", "attention_span"]
        }
        
        for section, fields in required_keys.items():
            if section not in data:
                raise AnalysisError(f"Missing required section: {section}")
            for field in fields:
                if field not in data[section]:
                    raise AnalysisError(f"Missing required field: {section}.{field}")

    def calculate_performance_metrics(self, 
                                   conversation_analysis: Dict[str, Any],
                                   quiz_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate various performance metrics
        
        Args:
            conversation_analysis: Analysis from analyze_conversation
            quiz_results: List of quiz performance records
            
        Returns:
            Dict containing performance metrics and their values
            
        Raises:
            MetricsCalculationError: If calculation fails
        """
        try:
            if not conversation_analysis:
                raise MetricsCalculationError("Empty conversation analysis provided")

            # Calculate quiz performance
            quiz_scores = [result.get('score_percentage', 0) for result in quiz_results if result]
            avg_quiz_score = statistics.mean(quiz_scores) if quiz_scores else 0
            
            # Calculate engagement score
            engagement_metrics = conversation_analysis.get('engagement_metrics', {})
            engagement_score = (
                engagement_metrics.get('participation_rate', 0) * 0.4 +
                engagement_metrics.get('response_quality', 0) * 0.4 +
                engagement_metrics.get('question_asking_frequency', 0) * 0.2
            )
            
            # Calculate understanding progression
            understanding_progress = conversation_analysis.get('understanding_progression', {})
            initial_level = understanding_progress.get('initial_level', 0)
            final_level = understanding_progress.get('final_level', 0)
            
            understanding_score = (
                (final_level - initial_level) /
                max(initial_level, 1) * 100
            ) if initial_level > 0 else 0
            
            return {
                "quiz_performance": avg_quiz_score,
                "engagement_quality": engagement_score,
                "concept_understanding": understanding_score,
                "progress_rate": (understanding_score + engagement_score) / 2
            }
            
        except statistics.StatisticsError as e:
            logger.error(f"Statistics calculation error: {str(e)}")
            raise MetricsCalculationError(f"Failed to calculate statistics: {str(e)}")
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise MetricsCalculationError(f"Failed to calculate metrics: {str(e)}")

    async def generate_final_report(self, 
                                  conversation_history: List[Dict[str, Any]], 
                                  quiz_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive final report
        
        Args:
            conversation_history: Complete conversation history
            quiz_results: List of quiz results
            
        Returns:
            Dict containing final report and visualization data
            
        Raises:
            AuditorError: If report generation fails
        """
        try:
            # Get conversation analysis
            conversation_analysis = await self.analyze_conversation(conversation_history)
            
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(conversation_analysis, quiz_results)
            
            # Calculate weighted total score
            total_score = sum(
                metric_value * self.weights[metric_name]
                for metric_name, metric_value in performance_metrics.items()
            )
            
            # Generate learning style tags
            learning_patterns = conversation_analysis.get('learning_patterns', {})
            learning_style = learning_patterns.get('preferred_learning_style', 'Unknown')
            effective_topics = learning_patterns.get('most_effective_topics', [])
            
            # Prepare visualization data
            pie_chart_data = {
                "metrics": [
                    {
                        "name": "Quiz Performance",
                        "percentage": round(performance_metrics['quiz_performance'], 2),
                        "weight": self.weights['quiz_performance']
                    },
                    {
                        "name": "Engagement Quality",
                        "percentage": round(performance_metrics['engagement_quality'], 2),
                        "weight": self.weights['engagement_quality']
                    },
                    {
                        "name": "Concept Understanding",
                        "percentage": round(performance_metrics['concept_understanding'], 2),
                        "weight": self.weights['concept_understanding']
                    },
                    {
                        "name": "Progress Rate",
                        "percentage": round(performance_metrics['progress_rate'], 2),
                        "weight": self.weights['progress_rate']
                    }
                ]
            }
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(performance_metrics, learning_patterns)
            
            return {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_score": round(total_score, 2),
                    "performance_level": self._get_performance_level(total_score)
                },
                "performance_metrics": performance_metrics,
                "learning_profile": {
                    "preferred_style": learning_style,
                    "effective_topics": effective_topics,
                    "attention_span": learning_patterns.get('attention_span', 'Unknown')
                },
                "progress_analysis": {
                    "initial_level": conversation_analysis['understanding_progression']['initial_level'],
                    "final_level": conversation_analysis['understanding_progression']['final_level'],
                    "key_improvements": conversation_analysis['understanding_progression']['key_improvements'],
                    "challenging_areas": conversation_analysis['understanding_progression']['challenging_areas']
                },
                "recommendations": recommendations,
                "visualization_data": pie_chart_data
            }
            
        except (AnalysisError, MetricsCalculationError) as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise AuditorError(f"Report generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in report generation: {str(e)}")
            raise AuditorError(f"Unexpected error in report generation: {str(e)}")

    async def _generate_recommendations(self, 
                                     performance_metrics: Dict[str, float], 
                                     learning_patterns: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate personalized recommendations based on performance"""
        try:
            messages = [
                SystemMessage(content="""You are an expert Course Auditor.
                Based on the student's performance metrics and learning patterns,
                provide specific recommendations for improvement."""),
                
                HumanMessage(content=f"""Performance Metrics:
                {json.dumps(performance_metrics, indent=2)}
                
                Learning Patterns:
                {json.dumps(learning_patterns, indent=2)}
                
                Provide targeted recommendations in this format:
                {{
                    "key_strengths": [str],
                    "improvement_areas": [str],
                    "action_items": [str],
                    "additional_resources": [str]
                }}""")
            ]
            
            response = await self.llm.ainvoke(messages)
            return json.loads(response.content)
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return {
                "key_strengths": ["Unable to analyze strengths due to error"],
                "improvement_areas": ["Unable to analyze improvement areas due to error"],
                "action_items": ["Please contact support for detailed recommendations"],
                "additional_resources": ["General learning resources"]
            }

    def _get_performance_level(self, total_score: float) -> str:
        """Determine performance level based on total score"""
        if not isinstance(total_score, (int, float)):
            raise ValueError("Invalid total score format")
            
        if total_score >= 90:
            return "Outstanding"
        elif total_score >= 80:
            return "Excellent"
        elif total_score >= 70:
            return "Good"
        elif total_score >= 60:
            return "Satisfactory"
        else:
            return "Needs Improvement"

async def generate_audit_report(conversation_history: List[Dict[str, Any]], 
                              quiz_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convenience function to generate an audit report
    
    Args:
        conversation_history: Complete conversation history
        quiz_results: List of quiz results
        
    Returns:
        Dict containing the audit report or None if generation fails
    """
    try:
        auditor = CourseAuditor()
        return await auditor.generate_final_report(conversation_history, quiz_results)
    except Exception as e:
        logger.error(f"Failed to generate audit report: {str(e)}")
        return None