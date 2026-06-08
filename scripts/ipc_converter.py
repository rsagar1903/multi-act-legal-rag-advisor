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
    # Remove markers like 1[...], 2[...], etc., but keep the content inside the brackets.
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
    Checks if a line is a marker for a state amendment block.
    """
    state_names = ["karnataka", "orissa", "chhattisgarh", "maharashtra", "himachal pradesh", "tripura", "jammu and kashmir"]
    # Check for state names, "STATE AMENDMENT", or citation markers
    return text.strip().lower() in state_names or \
           "STATE AMENDMENT" in text.upper() or \
           text.strip().startswith("[Vide")

def flatten_paragraphs(p_dict: dict) -> list[str]:
    """
    Recursively flattens the nested paragraph structure from the source JSON
    into a clean list of strings, filtering out state amendments.
    """
    parts = []
    if not isinstance(p_dict, dict):
        return parts

    try:
        # Sort keys numerically to maintain the original order of paragraphs
        sorted_keys = sorted(p_dict.keys(), key=int)
    except (ValueError, AttributeError):
        # Fallback for non-integer keys
        sorted_keys = sorted(p_dict.keys())

    for key in sorted_keys:
        value = p_dict[key]
        if isinstance(value, str):
            if not is_state_amendment_marker(value):
                cleaned_value = clean_text(value)
                # Avoid adding standalone headers that are handled elsewhere
                if cleaned_value and cleaned_value.lower() not in ["illustrations", "illustration", "explanation", "provisos", "proviso"]:
                    parts.append(cleaned_value)
        elif isinstance(value, dict):
            # If the dictionary has a 'text' key, process it
            if 'text' in value:
                text_part = clean_text(value['text'])
                if text_part and not is_state_amendment_marker(text_part):
                     parts.append(text_part)
            # Recursively process the 'contains' dictionary if it exists
            if 'contains' in value:
                parts.extend(flatten_paragraphs(value['contains']))
    return parts

def format_section_id(raw_id_str: str) -> str:
    """
    Formats a raw section identifier (e.g., '1', '52A') into a
    standardized format ('001', '052A').
    """
    if raw_id_str.isdigit():
        return raw_id_str.zfill(3)
    
    match = re.match(r'(\d+)([A-Z]+)', raw_id_str, re.I)
    if match:
        num_part, letter_part = match.groups()
        return num_part.zfill(3) + letter_part.upper()
    return raw_id_str

def process_section(section_key: str, section_data: dict, chapter_name: str, index: int) -> dict:
    """
    Processes a single section from the IPC JSON and converts it into the
    target BNS chunk format.
    """
    # 1. Extract and format the section number
    raw_num_match = re.search(r'(\d+[A-Z]?)', section_key)
    if not raw_num_match:
        return None  # Skip if no valid section number is found
    raw_num = raw_num_match.group(1)
    section_id = format_section_id(raw_num)

    # 2. Extract and clean the heading
    heading = section_data.get('heading', 'No Heading').strip().rstrip('.')

    # 3. Flatten and join paragraphs to form the raw content
    paragraphs = section_data.get('paragraphs', {})
    content_parts = flatten_paragraphs(paragraphs)
    raw_content = " ".join(part for part in content_parts if part)

    # 4. Construct the formatted `content` string
    content = (
        f"IPC Section {section_id}: {heading}\n"
        f"Chapter: {chapter_name}\n\n"
        f"Legal Text:\n{raw_content}\n"
        f"(End of Section {section_id})"
    )

    # 5. Assemble and return the final dictionary object
    chunk = {
        "id": f"ipc_{section_id.lower()}_{index}",
        "section": section_id,
        "heading": heading,
        "content": content,
        "raw_content": raw_content,
        "chapter": chapter_name,
        "section_display": f"Section {section_id}"
    }
    return chunk

def convert_ipc_to_bns_format(source_file: str, output_file: str):
    """
    Main function to read the source IPC JSON, convert it, and write the
    output to a new JSON file.
    """
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            ipc_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at '{source_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{source_file}'")
        return

    ipc_chunks = []
    index_counter = 0

    # Iterate through all chapters
    for chapter_key in ipc_data.get('Chapters', {}):
        chapter = ipc_data['Chapters'][chapter_key]
        chapter_name = chapter.get('Name', 'Unnamed Chapter')

        # Process sections directly under the chapter
        if 'Sections' in chapter:
            for section_key, section_data in chapter['Sections'].items():
                chunk = process_section(section_key, section_data, chapter_name, index_counter)
                if chunk:
                    ipc_chunks.append(chunk)
                    index_counter += 1

        # Process sections within subheadings of the chapter
        if 'Subheadings' in chapter:
            for subheading in chapter['Subheadings']:
                if 'Sections' in subheading:
                    for section_key, section_data in subheading['Sections'].items():
                        chunk = process_section(section_key, section_data, chapter_name, index_counter)
                        if chunk:
                            ipc_chunks.append(chunk)
                            index_counter += 1

    # Write the converted data to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ipc_chunks, f, indent=2, ensure_ascii=False)
        print(f"Successfully converted {len(ipc_chunks)} sections.")
        print(f"Output saved to '{os.path.abspath(output_file)}'")
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}")


# --- Main Execution ---
if __name__ == "__main__":
    SOURCE_JSON_FILE = 'data/A1860-45.json'
    OUTPUT_JSON_FILE = 'data/ipc_chunks.json'
    
    convert_ipc_to_bns_format(SOURCE_JSON_FILE, OUTPUT_JSON_FILE)