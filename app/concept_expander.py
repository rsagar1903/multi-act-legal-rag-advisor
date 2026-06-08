import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

OFFENSE_CACHE = {
    "theft": ["larceny", "stealing"],
    "riot": ["unlawful assembly", "mob violence"]
}

def expand_offenses(offense_list: list) -> list:
    """Generate legal synonyms for offenses"""
    expanded = set()
    
    for offense in offense_list:
        offense_lower = offense.lower()
        if offense_lower in OFFENSE_CACHE:
            expanded.update(OFFENSE_CACHE[offense_lower])
            continue
            
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{
                    "role": "system",
                    "content": f"Generate 2-3 alternative legal terms for '{offense}'"
                }],
                temperature=0.5,
                max_tokens=30
            )
            variations = [offense] + [
                v.strip('"').strip() for v in 
                response.choices[0].message.content.split("\n") 
                if v.strip()
            ]
            expanded.update(variations)
        except Exception as e:
            print(f"Error expanding '{offense}': {str(e)}")
            expanded.add(offense)
    
    return list(expanded)