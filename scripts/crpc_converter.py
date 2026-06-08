import json
import re
import os

def clean_text(text: str) -> str:
    """
    Cleans the legal text by removing footnote markers, bracketed numbers,
    and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Remove markers like 1[...], 2[...], etc., but keep the content.
    text = re.sub(r'\d+\[', '', text)
    text = text.replace(']', '')
    # Remove markers like 1***, 2* * *, etc.
    text = re.sub(r'\s*\d+\*[\s\*]*', '', text)
    # Remove any remaining standalone footnote numbers like [1]
    text = re.sub(r'\[\d+\]', '', text)
    # Collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_state_amendment_marker(text: str) -> bool:
    """
    Checks if a line is a marker for a state amendment block to be excluded.
    """
    text_lower = text.strip().lower()
    state_names = [
        "karnataka", "orissa", "chhattisgarh", "maharashtra", 
        "himachal pradesh", "tripura", "jammu and kashmir", "haryana",
        "west bengal", "gujarat", "madhya pradesh", "assam", "delhi",
        "andhra pradesh", "union territories of andaman and nicobar islands",
        "lakshadweep", "dadra and nagar haveli"
    ]
    
    # Check for state names, "STATE AMENDMENT", or citation markers
    if "STATE AMENDMENT" in text.upper() or text.strip().startswith("[Vide"):
        return True
    for state in state_names:
        if text_lower.startswith(state):
            return True
            
    return False

def flatten_paragraphs(p_dict: dict) -> list[str]:
    """
    Recursively flattens the nested paragraph structure from the source JSON
    into a clean list of strings, filtering out state amendments.
    """
    parts = []
    if not isinstance(p_dict, dict):
        return parts

    try:
        sorted_keys = sorted(p_dict.keys(), key=int)
    except (ValueError, AttributeError):
        sorted_keys = sorted(p_dict.keys())

    for key in sorted_keys:
        value = p_dict[key]
        if isinstance(value, str):
            if not is_state_amendment_marker(value):
                cleaned_value = clean_text(value)
                if cleaned_value:
                    parts.append(cleaned_value)
        elif isinstance(value, dict):
            if 'text' in value:
                text_part = clean_text(value['text'])
                if text_part and not is_state_amendment_marker(text_part):
                     parts.append(text_part)
            if 'contains' in value:
                parts.extend(flatten_paragraphs(value['contains']))
    return parts

def format_section_id(raw_id_str: str) -> str:
    """
    Formats a raw section identifier (e.g., '1', '41A') into a
    standardized format ('001', '041A').
    """
    if raw_id_str.isdigit():
        return raw_id_str.zfill(3)
    
    match = re.match(r'(\d+)([A-Z]+)', raw_id_str, re.I)
    if match:
        num_part, letter_part = match.groups()
        return num_part.zfill(3) + letter_part.upper()
    return raw_id_str

def process_section(section_key: str, section_data: dict, chapter_name: str, index: int, act_info: dict) -> dict:
    """
    Processes a single section from the source JSON and converts it into the
    target chunk format.
    """
    raw_num_match = re.search(r'(\d+[A-Z]?)', section_key)
    if not raw_num_match:
        return None
    raw_num = raw_num_match.group(1)
    section_id = format_section_id(raw_num)

    heading = section_data.get('heading', 'No Heading').strip().rstrip('.')

    paragraphs = section_data.get('paragraphs', {})
    content_parts = flatten_paragraphs(paragraphs)
    raw_content = " ".join(part for part in content_parts if part)
    
    if not raw_content:
        return None # Skip sections that are empty after filtering (e.g., repealed sections)

    content = (
        f"{act_info['prefix']} Section {section_id}: {heading}\n"
        f"Chapter: {chapter_name}\n\n"
        f"Legal Text:\n{raw_content}\n"
        f"(End of Section {section_id})"
    )

    chunk = {
        "id": f"{act_info['id']}_{section_id.lower()}_{index}",
        "section": section_id,
        "heading": heading,
        "content": content,
        "raw_content": raw_content,
        "chapter": chapter_name,
        "section_display": f"Section {section_id}"
    }
    return chunk

def convert_law_to_chunks(source_file: str, output_file: str, act_info: dict):
    """
    Main function to read a source law JSON, convert it, and write the output.
    """
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at '{source_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{source_file}'")
        return

    law_chunks = []
    index_counter = 0

    for chapter_key, chapter in sorted(source_data.get('Chapters', {}).items()):
        chapter_name = chapter.get('Name', '').strip()
        if not chapter_name: # Fallback for chapters with no name
            chapter_name = chapter.get('ID', f'Unnamed Chapter {chapter_key}').strip()

        if 'Sections' in chapter:
            for section_key, section_data in chapter['Sections'].items():
                chunk = process_section(section_key, section_data, chapter_name, index_counter, act_info)
                if chunk:
                    law_chunks.append(chunk)
                    index_counter += 1

        if 'Subheadings' in chapter:
            for subheading in chapter['Subheadings']:
                if 'Sections' in subheading:
                    for section_key, section_data in subheading['Sections'].items():
                        chunk = process_section(section_key, section_data, chapter_name, index_counter, act_info)
                        if chunk:
                            law_chunks.append(chunk)
                            index_counter += 1

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(law_chunks, f, indent=2, ensure_ascii=False)
        print(f"Successfully converted {len(law_chunks)} sections from {source_file}.")
        print(f"Output saved to '{os.path.abspath(output_file)}'")
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Define parameters for the CrPC file
    crpc_source_file = 'data/1974__02.json'
    crpc_output_file = 'data/crpc_chunks.json'
    crpc_act_info = {
        'id': 'crpc',
        'prefix': 'CrPC'
    }
    
    # Run the conversion
    convert_law_to_chunks(crpc_source_file, crpc_output_file, crpc_act_info)