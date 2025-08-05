import os
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_supabase() -> Client:
    """Initialize Supabase client using environment variables."""
    try:
        # Check if .env file exists and is in the correct location
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            logger.error(f"No .env file found at {env_path}")
            raise ValueError("Missing .env file. Create one from .env.example and add your Supabase credentials.")
            
        # Load environment variables from the specific path
        load_dotenv(dotenv_path=env_path)
        
        # Get Supabase credentials from environment
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        # Validate credentials with detailed messages
        if not supabase_url and not supabase_key:
            raise ValueError("Both SUPABASE_URL and SUPABASE_KEY are missing in .env file")
        elif not supabase_url:
            raise ValueError("SUPABASE_URL is missing in .env file")
        elif not supabase_key:
            raise ValueError("SUPABASE_KEY is missing in .env file")
        elif supabase_url == "your_supabase_project_url" or supabase_key == "your_supabase_anon_key":
            raise ValueError("Please replace the placeholder values in .env with your actual Supabase credentials")
            
            try:
                # Initialize Supabase client
                supabase: Client = create_client(supabase_url, supabase_key)
                
                # Test connection with proper error handling
                try:
                    # Execute a simple query to test the connection
                    response = supabase.table('exams').select("count", count='exact').execute()
                    
                    # Verify response format
                    if isinstance(response, dict) and 'data' in response:
                        logger.info("✓ Supabase client initialized and connected successfully")
                        return supabase
                    else:
                        logger.error(f"Unexpected response format: {type(response)}")
                        raise ValueError("Invalid response format from Supabase")            except Exception as e:
                logger.error(f"Failed to test Supabase connection: {str(e)}")
                if "404" in str(e):
                    raise ValueError("Table 'exams' not found. Please ensure your database is properly set up.")
                elif "401" in str(e):
                    raise ValueError("Authentication failed. Please check your Supabase credentials.")
                elif "Network Error" in str(e):
                    raise ValueError("Network error. Please check your internet connection.")
                raise
                
        except Exception as e:
            logger.error(f"Supabase client creation failed: {str(e)}")
            raise ValueError(f"Failed to create Supabase client: {str(e)}")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return None

# Global Supabase client instance
SUPABASE_CLIENT = initialize_supabase()
