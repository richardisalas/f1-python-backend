import os
import logging
import json
import argparse
import sys
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import re

# Load environment variables
load_dotenv()

# Set up OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1Agent:
    def __init__(self, config: Optional[Dict[str, Any]] = None, web_search: bool = True, rag: bool = False):
        self.config = config or {}
        self.memory = []
        self.web_search = web_search
        self.rag = rag
        logger.info(f"F1 Agent initialized with web_search={web_search}, rag={rag}")
    
    def perceive(self, input_data: Any) -> Dict[str, Any]:
        """Process input data from the environment"""
        logger.info("Processing input data")
        # Process and structure the input data
        return {"processed_data": input_data}
    
    def think(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reason about the processed data and decide on action using AI with optional web search or RAG"""
        if self.rag:
            logger.info("Thinking about data with AI and RAG")
        else:
            logger.info(f"Thinking about data with AI {'and web search' if self.web_search else ''}")
        
        # Extract query
        data = processed_data.get("processed_data", {})
        query = data.get("query", "")
        
        try:
            if self.rag:
                # Use standard model for RAG (currently same as no web search)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user", 
                            "content": f"As an F1 expert, please answer this question using RAG: {query}"
                        }
                    ],
                )
                
                # Extract the response content
                ai_response = response.choices[0].message.content
                
                # Format response as our standard format
                result = {
                    "decision": "rag_response",
                    "params": {},
                    "reasoning": "Used RAG to answer F1 information query",
                    "result": ai_response
                }
            elif self.web_search:
                # Use the web search capability with the correct model and syntax
                response = client.chat.completions.create(
                    model="gpt-4o-search-preview",
                    web_search_options={},
                    messages=[
                        {
                            "role": "user", 
                            "content": f"As an F1 expert, please answer this question with the most up-to-date information: {query}"
                        }
                    ],
                )
                
                # Extract the response content
                ai_response = response.choices[0].message.content
                
                # Extract any citations if present
                citations = []
                if hasattr(response.choices[0].message, 'annotations'):
                    for annotation in response.choices[0].message.annotations:
                        if annotation.type == "url_citation":
                            citations.append({
                                "title": annotation.url_citation.title,
                                "url": annotation.url_citation.url
                            })
                
                # Format response as our standard format
                result = {
                    "decision": "web_search_response",
                    "params": {"citations": citations},
                    "reasoning": "Searched the web for up-to-date F1 information",
                    "result": ai_response
                }
            else:
                # Use standard model without web search
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user", 
                            "content": f"As an F1 expert, please answer this question: {query}"
                        }
                    ],
                )
                
                # Extract the response content
                ai_response = response.choices[0].message.content
                
                # Format response as our standard format
                result = {
                    "decision": "ai_response",
                    "params": {},
                    "reasoning": "Used AI model to answer F1 information query",
                    "result": ai_response
                }
            
            return result
            
        except Exception as e:
            logger.error(f"AI thinking error: {e}")
            # Fallback to basic decision
            return {"decision": "fallback_action", "params": {}, "error": str(e), "result": f"I encountered an error while searching for information: {str(e)}"}
    
    def _construct_prompt(self, processed_data: Dict[str, Any]) -> str:
        """Construct a prompt for the AI based on the processed data"""
        # Convert processed data to a readable format
        data_str = json.dumps(processed_data, indent=2)
        
        # Build context from memory if available
        context = ""
        if self.memory and len(self.memory) > 0:
            last_interactions = self.memory[-3:] if len(self.memory) > 3 else self.memory
            context = "Previous interactions:\n" + json.dumps(last_interactions, indent=2) + "\n\n"
        
        # Construct the full prompt
        prompt = f"""
You are an F1 race strategy AI assistant. Based on the following information, 
provide analysis and recommendations:

{context}
Current input data:
{data_str}

Respond with a JSON object containing:
1. A 'decision' field with the recommended action
2. A 'params' object with parameters for the action
3. A 'reasoning' field explaining your analysis
4. A 'result' field with a user-friendly explanation
"""
        return prompt
    
    def _parse_ai_response(self, ai_response: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the AI response into a structured format"""
        try:
            # Try to extract JSON from the response
            # Look for JSON block in the response
            if "```json" in ai_response and "```" in ai_response.split("```json", 1)[1]:
                json_str = ai_response.split("```json", 1)[1].split("```", 1)[0].strip()
                result = json.loads(json_str)
            elif "{" in ai_response and "}" in ai_response:
                # Extract the JSON object from the text
                json_str = ai_response[ai_response.find("{"):ai_response.rfind("}")+1]
                result = json.loads(json_str)
            else:
                # Use the full text as reasoning if no JSON found
                result = {
                    "decision": "ai_response",
                    "params": {},
                    "reasoning": ai_response,
                    "result": ai_response
                }
                
            # Ensure all required fields exist
            if "decision" not in result:
                result["decision"] = "analyze"
            if "params" not in result:
                result["params"] = {}
            if "reasoning" not in result:
                result["reasoning"] = ai_response
            if "result" not in result:
                result["result"] = "Analysis complete"
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return {
                "decision": "parse_error",
                "params": {},
                "reasoning": "Error parsing AI response",
                "result": ai_response,
                "original_data": original_data
            }
    
    def act(self, decision: Dict[str, Any]) -> Any:
        """Execute actions based on decisions"""
        logger.info(f"Taking action: {decision['decision']}")
        
        # Extract the result or reasoning from the decision
        result = decision.get("result", "")
        if not result and "reasoning" in decision:
            result = decision["reasoning"]
        
        return {
            "status": "success", 
            "result": result
        }
    
    def run_step(self, input_data: Any) -> Any:
        """Run a single step of the agent loop"""
        processed_data = self.perceive(input_data)
        decision = self.think(processed_data)
        result = self.act(decision)
        self.memory.append({"input": input_data, "output": result})
        return result

def get_user_input() -> Dict[str, Any]:
    """Get input from the user via simple chat interface"""
    try:
        user_message = input("\nYou: ").strip()
        
        if not user_message:
            return {}
            
        if user_message.lower() in ["exit", "quit", "bye"]:
            return {}
            
        # Convert user message to input data
        return {
            "query": user_message,
            "type": "chat"
        }
        
    except KeyboardInterrupt:
        print("\nInput cancelled")
        return {}
    except Exception as e:
        print(f"\nError during input: {e}")
        return {}

def show_help(web_search: bool = True, rag: bool = False):
    """Display help information"""
    print("\n=== F1 Agent Help ===")
    print("This tool helps analyze F1 race data and provide strategies.")
    print("\nOptions:")
    print("1. Enter race information - Input details about a race")
    print("2. Query historical data - Look up past races and statistics")
    print("3. Get strategy recommendations - Get tire and pit strategies")
    print("4. Help - Show this help information")
    print("0. Exit - Close the application")
    print("\nFeatures:")
    print("- Web search is " + ("ENABLED" if web_search else "DISABLED") + " (use --no-web-search to disable)")
    print("- RAG is " + ("ENABLED" if rag else "DISABLED") + " (use --rag to enable)")
    print("\nTips:")
    print("- You don't need to fill every prompt, just press Enter to skip")
    print("- Additional information can be added for more specific analysis")
    print("- The agent will remember past interactions during your session")
    input("\nPress Enter to continue...")

def setup_cli_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="F1 AI Agent for data analysis and strategy optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--log-level",
        help="Set logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Input data as JSON string or path to JSON file",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--batch", "-b",
        help="Process input without interactive mode",
        action="store_true"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path for results",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--no-web-search",
        help="Disable web search capability (uses gpt-4o instead of gpt-4o-search-preview)",
        action="store_true"
    )
    
    parser.add_argument(
        "--rag",
        help="Enable Retrieval-Augmented Generation",
        action="store_true"
    )
    
    return parser

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def load_input_data(input_string: str) -> Dict[str, Any]:
    """Load input data from string or file"""
    # Check if input is a file path
    if os.path.exists(input_string):
        try:
            with open(input_string, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input file: {e}")
            return {}
    
    # Otherwise treat as JSON string
    try:
        return json.loads(input_string)
    except json.JSONDecodeError:
        # Try key=value format
        result = {}
        pairs = input_string.split(",")
        for pair in pairs:
            if "=" in pair:
                key, value = pair.strip().split("=", 1)
                result[key.strip()] = value.strip()
        return result

def save_output(data: Any, output_path: str) -> bool:
    """Save output data to file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        return False

def main():
    # Parse command line arguments
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Initialize the agent
    agent = F1Agent(config=config, web_search=not args.no_web_search, rag=args.rag)
    
    # Batch mode
    if args.batch and args.input:
        input_data = load_input_data(args.input)
        result = agent.run_step(input_data)
        
        if args.output:
            save_output(result, args.output)
        else:
            print(json.dumps(result, indent=2))
        
        return
    
    # Interactive mode
    try:
        print("====================================")
        print("F1 Assistant - Chat Interface")
        print("====================================")
        print("Welcome! Ask any questions about F1 racing, strategy, or historical data.")
        print(f"Web search is {'ENABLED' if not args.no_web_search else 'DISABLED'}")
        print(f"RAG is {'ENABLED' if args.rag else 'DISABLED'}")
        print("Type 'exit' or press Ctrl+C to quit.")
        
        while True:
            # Get user input
            input_data = get_user_input()
            
            # Exit if empty input
            if not input_data:
                print("\nThank you for using F1 Assistant. Goodbye!")
                break
            
            # Run a single step
            try:
                result = agent.run_step(input_data)
                
                # Present results in a more user-friendly way
                if "result" in result and result["result"]:
                    # Clean response of all citations and URLs
                    clean_response = result["result"]
                    clean_response = re.sub(r'\[\d+\]', '', clean_response)  # Remove numbered citations
                    clean_response = re.sub(r'\[([^\]]+?\.[a-z]{2,}[^\]]*?)\]', '', clean_response)  # Remove URL citations
                    clean_response = re.sub(r'\[([^\]]+?)\]\([^)]+?\)', r'\1', clean_response)  # Replace markdown links with just text
                    clean_response = re.sub(r'https?://\S+', '', clean_response)  # Remove any remaining URLs
                    
                    # Remove text between "((" markers and the markers themselves
                    clean_response = re.sub(r'\(\([^(]*?(?=\(\(|\Z)', '', clean_response)  # Remove text between (( and the next (( or end
                    clean_response = re.sub(r'\(\(', '', clean_response)  # Remove any remaining (( characters
                    
                    # Remove "Recent Developments" section if present
                    if "## Recent Developments" in clean_response:
                        clean_response = clean_response.split("## Recent Developments")[0].strip()
                        
                    print(f"\nAssistant: {clean_response}")
                else:
                    # Fallback to showing a generic message
                    print("\nAssistant: I'm sorry, I couldn't process that request.")
                    
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                print(f"\nAssistant: Sorry, I encountered an error: {e}")
    
    except KeyboardInterrupt:
        print("\nInput cancelled")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
