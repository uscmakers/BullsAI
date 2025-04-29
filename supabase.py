import os
import time

from dotenv import load_dotenv

from supabase import create_client

# Load environment variables from .env file
load_dotenv()

# Get Supabase credentials from environment variables
supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# Create Supabase client
supabase = create_client(supabase_url, supabase_key)

def add_dart_score(player, score, round_number):
    """
    Add a dart score to the database
    
    Args:
        player (str): Either "robot" or "human"
        score (int): The score for this throw
        round_number (int): The current round number
        
    Returns:
        dict: The inserted data or error information
    """
    try:
        # Insert the score into the dart_scores table
        data, error = supabase.table("dart_scores").insert({
            "player": player,
            "score": score,
            "round": round_number,
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%S%z')
        }).execute()
        
        if error:
            return {"success": False, "error": error}
        
        return {"success": True, "data": data}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    
if __name__ == "__main__":
    # Example: Add a robot score
    result = add_dart_score("robot", 20, 1)
    print(result)
    
    # Example: Add a human score
    result = add_dart_score("human", 15, 1)
    print(result)