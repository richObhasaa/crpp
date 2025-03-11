import os
import json
import re
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL_NAME = "gpt-4o"

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai = OpenAI(api_key=OPENAI_API_KEY)

def analyze_whitepaper(whitepaper_text):
    """
    Analyze a cryptocurrency whitepaper using OpenAI GPT-4o
    
    Args:
        whitepaper_text (str): The text content of the whitepaper
        
    Returns:
        dict: Analysis results including summary, technical evaluation, risk assessment, etc.
    """
    try:
        if not OPENAI_API_KEY:
            return {
                "error": "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            }
        
        if not whitepaper_text or len(whitepaper_text.strip()) < 100:
            return {
                "error": "Whitepaper text is too short or empty. Please provide a valid whitepaper."
            }
        
        # Limit the whitepaper text length if it's too long
        max_text_length = 15000  # Approximate limit to stay within token limits
        if len(whitepaper_text) > max_text_length:
            whitepaper_text = whitepaper_text[:max_text_length] + "...\n[Content truncated due to length]"
        
        # Define the prompt for whitepaper analysis
        system_prompt = """
        You are a cryptocurrency and blockchain expert analyzing a whitepaper. Provide a comprehensive analysis with the following sections:

        1. Executive Summary (3-5 sentences overview)
        2. Technology Analysis (evaluation of the blockchain technology, consensus mechanism, scalability, etc.)
        3. Use Case & Value Proposition (assessment of the actual use case and value proposition)
        4. Tokenomics Analysis (evaluation of token economics, distribution, utility)
        5. Team & Development Assessment (if information available)
        6. Risk Assessment (potential risks and challenges)
        7. Market Potential (market opportunity and competitiveness)
        8. Technical Feasibility Rating (1-10 scale with justification)
        9. Investment Outlook (short, medium, and long-term potential)
        10. Key Strengths & Weaknesses (bullet points)

        Format your response as JSON with these sections as keys. Be objective, informative, and analytical.
        """
        
        user_prompt = f"Please analyze the following cryptocurrency whitepaper:\n\n{whitepaper_text}"
        
        # Make the API call
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=1500
        )
        
        analysis_result = json.loads(response.choices[0].message.content)
        
        # Add metadata to the result
        analysis_result["analysis_timestamp"] = response.created
        analysis_result["model_used"] = MODEL_NAME
        
        return analysis_result
    except json.JSONDecodeError:
        # If the response isn't valid JSON, try to extract the content more flexibly
        try:
            content = response.choices[0].message.content
            # Try to extract a JSON object using regex
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                analysis_result = json.loads(json_match.group(1))
                analysis_result["analysis_timestamp"] = response.created
                analysis_result["model_used"] = MODEL_NAME
                return analysis_result
            else:
                return {
                    "error": "Failed to parse the analysis result as JSON",
                    "raw_content": content
                }
        except Exception as inner_e:
            return {
                "error": f"Error processing analysis result: {str(inner_e)}"
            }
    except Exception as e:
        return {
            "error": f"Error analyzing whitepaper: {str(e)}"
        }

def extract_key_metrics_from_whitepaper(whitepaper_text):
    """
    Extract key metrics and figures from a cryptocurrency whitepaper
    
    Args:
        whitepaper_text (str): The text content of the whitepaper
        
    Returns:
        dict: Key metrics extracted from the whitepaper
    """
    try:
        if not OPENAI_API_KEY:
            return {
                "error": "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            }
        
        if not whitepaper_text or len(whitepaper_text.strip()) < 100:
            return {
                "error": "Whitepaper text is too short or empty. Please provide a valid whitepaper."
            }
        
        # Limit the whitepaper text length if it's too long
        max_text_length = 15000  # Approximate limit to stay within token limits
        if len(whitepaper_text) > max_text_length:
            whitepaper_text = whitepaper_text[:max_text_length] + "...\n[Content truncated due to length]"
        
        # Define the prompt for extracting key metrics
        system_prompt = """
        You are a data extraction specialist. Extract the following key metrics and figures from the cryptocurrency whitepaper:

        1. Total Token Supply
        2. Token Distribution Percentages
        3. Token Utility
        4. Transaction Speed (TPS)
        5. Consensus Mechanism
        6. Block Time
        7. Development Timeline
        8. Key Partners or Integrations
        9. Target Market Size
        10. Fee Structure

        For each metric, extract the specific value or description from the whitepaper. If a metric is not mentioned, indicate "Not specified in the whitepaper."
        Format your response as JSON with these metrics as keys.
        """
        
        user_prompt = f"Please extract key metrics and figures from the following cryptocurrency whitepaper:\n\n{whitepaper_text}"
        
        # Make the API call
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000
        )
        
        metrics_result = json.loads(response.choices[0].message.content)
        
        # Add metadata to the result
        metrics_result["extraction_timestamp"] = response.created
        metrics_result["model_used"] = MODEL_NAME
        
        return metrics_result
    except Exception as e:
        return {
            "error": f"Error extracting metrics from whitepaper: {str(e)}"
        }

def compare_whitepapers(whitepaper1_text, whitepaper2_text, project1_name, project2_name):
    """
    Compare two cryptocurrency whitepapers
    
    Args:
        whitepaper1_text (str): Text of the first whitepaper
        whitepaper2_text (str): Text of the second whitepaper
        project1_name (str): Name of the first project
        project2_name (str): Name of the second project
        
    Returns:
        dict: Comparison results
    """
    try:
        if not OPENAI_API_KEY:
            return {
                "error": "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            }
        
        # Validate inputs
        if not whitepaper1_text or not whitepaper2_text:
            return {
                "error": "Both whitepapers must be provided for comparison."
            }
        
        # Limit the whitepaper text length if it's too long
        max_text_length = 10000  # Smaller limit since we're sending two papers
        
        whitepaper1_truncated = whitepaper1_text
        whitepaper2_truncated = whitepaper2_text
        
        if len(whitepaper1_text) > max_text_length:
            whitepaper1_truncated = whitepaper1_text[:max_text_length] + "...\n[Content truncated due to length]"
        
        if len(whitepaper2_text) > max_text_length:
            whitepaper2_truncated = whitepaper2_text[:max_text_length] + "...\n[Content truncated due to length]"
        
        # Define the prompt for whitepaper comparison
        system_prompt = f"""
        You are a cryptocurrency and blockchain expert comparing two whitepapers. Provide a detailed comparison between {project1_name} and {project2_name} with the following sections:

        1. Technology Approach: Compare the technical approaches and architectures
        2. Use Cases: Compare the intended use cases and problem solutions
        3. Tokenomics: Compare token economics, distribution, and utility
        4. Consensus Mechanisms: Compare how consensus is achieved
        5. Scalability Solutions: Compare approaches to scaling
        6. Governance Models: Compare how decisions are made
        7. Security Approaches: Compare security measures and protections
        8. Relative Strengths: What each does better than the other
        9. Market Positioning: How they position themselves in the market
        10. Overall Assessment: Which whitepaper presents a more robust and feasible project

        Format your response as JSON with these sections as keys. Be objective, analytical, and specific in your comparisons.
        """
        
        user_prompt = f"""
        Please compare these two cryptocurrency whitepapers:
        
        {project1_name} Whitepaper:
        {whitepaper1_truncated}
        
        {project2_name} Whitepaper:
        {whitepaper2_truncated}
        """
        
        # Make the API call
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=1500
        )
        
        comparison_result = json.loads(response.choices[0].message.content)
        
        # Add metadata to the result
        comparison_result["comparison_timestamp"] = response.created
        comparison_result["model_used"] = MODEL_NAME
        comparison_result["projects_compared"] = [project1_name, project2_name]
        
        return comparison_result
    except Exception as e:
        return {
            "error": f"Error comparing whitepapers: {str(e)}"
        }
