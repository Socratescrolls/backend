import os
from dotenv import load_dotenv
def setup_environment():
    """Setup and validate environment variables with debug information"""
    print("Starting environment setup...")
    
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print(".env file found")
        try:
            with open('.env', 'r') as f:
                print("Contents of .env file:")
                print(f.read())
        except Exception as e:
            print(f"Error reading .env file: {e}")
    else:
        print(".env file not found!")
        print("Looking for .env in:", os.getcwd())
    
    # Load environment variables
    print("\nLoading environment variables...")
    load_dotenv()
    
    # Debug: Print relevant environment variables
    print("\nEnvironment variables after load_dotenv:")
    print(f"LANGCHAIN_API_KEY present: {'LANGCHAIN_API_KEY' in os.environ}")
    print(f"OPENAI_API_KEY2 present: {'OPENAI_API_KEY2' in os.environ}")
    
    # Configure API keys and settings
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    # Get API keys with debug output
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY2")
    
    print("\nAPI Keys status:")
    print(f"LANGCHAIN_API_KEY: {'Found' if langchain_key else 'Missing'}")
    print(f"OPENAI_API_KEY2: {'Found' if openai_key else 'Missing'}")
    
    if not langchain_key or not openai_key:
        missing_keys = []
        if not langchain_key:
            missing_keys.append("LANGCHAIN_API_KEY")
        if not openai_key:
            missing_keys.append("OPENAI_API_KEY2")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
    os.environ["OPENAI_API_KEY2"] = openai_key
    
    print("\nEnvironment setup completed successfully!")

setup_environment()