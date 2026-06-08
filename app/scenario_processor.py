import openai
import json
from dotenv import load_dotenv
import os
try:
    from .agent_router import detect_acts_from_query
except ImportError:
    from agent_router import detect_acts_from_query

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_scenario(scenario: str) -> dict:
    """Enhanced scenario analysis that suggests relevant acts"""
    prompt = f"""Analyze this legal scenario and suggest:
    1. PRIMARY criminal/civil offense (most serious applicable)
    2. RELATED offenses (other applicable charges)
    3. RELEVANT LEGAL ACTS (BNS/IPC/CRPC/CPC/BSA)
    4. KEY ELEMENTS needed to prove the case

    Scenario: {scenario}

    Output JSON format:
    {{
        "primary_offense": "...",
        "related_offenses": ["...", "..."],
        "relevant_acts": ["bns", "ipc", ...],
        "key_elements": ["...", "..."]
    }}"""
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{
            "role": "system",
            "content": prompt
        },{
            "role": "user",
            "content": scenario
        }],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    result = json.loads(response.choices[0].message.content)
    
    # Fallback: if LLM doesn't specify acts, detect from scenario
    if not result.get("relevant_acts"):
        result["relevant_acts"] = detect_acts_from_query(scenario)
    
    return result
