import pandas as pd
import json
import os
import re

def format_section_id(raw_id) -> str:
    """
    Formats a raw section identifier (e.g., 1, '52A') into a
    standardized string format ('001', '052A').
    """
    try:
        # Convert numbers (like 1.0 from pandas) to int, then to string
        raw_id_str = str(int(float(raw_id)))
    except (ValueError, TypeError):
        raw_id_str = str(raw_id) # Handle non-numeric cases like '52A'

    if raw_id_str.isdigit():
        return raw_id_str.zfill(3)
    
    match = re.match(r'(\d+)([A-Z]+)', raw_id_str, re.I)
    if match:
        num_part, letter_part = match.groups()
        return num_part.zfill(3) + letter_part.upper()
    
    return raw_id_str

def convert_bsa_csv_to_json(csv_file: str, json_file: str):
    """
    Reads a CSV with specific BSA headers and converts it into a structured JSON file.
    """
    try:
        df = pd.read_csv(csv_file)
        df.dropna(how='all', inplace=True)
    except FileNotFoundError:
        print(f"Error: Source file not found at '{csv_file}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Clean column names to remove leading/trailing spaces (like in 'Section _name ')
    df.columns = df.columns.str.strip()

    # Forward-fill Chapter and Chapter_name columns for sections under the same chapter
    # This handles cases where the chapter name is only listed once for multiple sections.
    df[['Chapter', 'Chapter_name']] = df[['Chapter', 'Chapter_name']].ffill()
        
    bsa_chunks = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Map the correct CSV headers to our variables
        section_raw = row.get('Section')
        chapter_name = str(row.get('Chapter_name', 'N/A')).strip()
        chapter_subtype = str(row.get('Chapter_subtype', '')).strip()
        # **Using 'Section_name' as the heading**
        heading = str(row.get('Section _name', 'No Heading')).strip()
        raw_content = str(row.get('Description', '')).strip()

        # Skip rows that don't have a section number
        if not section_raw or pd.isna(section_raw):
            continue
            
        # Combine Chapter and Sub-type for a more descriptive chapter field
        full_chapter_name = chapter_name
        if chapter_subtype and chapter_subtype.lower() != 'nan':
            full_chapter_name = f"{chapter_name}: {chapter_subtype}"

        # Format the section number
        section_id = format_section_id(section_raw)

        # Construct the formatted 'content' field
        content = (
            f"BSA Section {section_id}: {heading}\n"
            f"Chapter: {full_chapter_name}\n\n"
            f"Legal Text:\n{raw_content}\n"
            f"(End of Section {section_id})"
        )

        # Assemble the final dictionary for the current section
        chunk = {
            "id": f"bsa_{section_id.lower()}_{index}",
            "section": section_id,
            "heading": heading,
            "content": content,
            "raw_content": raw_content,
            "chapter": full_chapter_name,
            "section_display": f"Section {section_id}"
        }
        
        bsa_chunks.append(chunk)

    # Write the list of dictionaries to the output JSON file
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(bsa_chunks, f, indent=2, ensure_ascii=False)
        print(f"Successfully converted {len(bsa_chunks)} sections from '{csv_file}'.")
        print(f"Output saved to '{os.path.abspath(json_file)}'")
    except IOError as e:
        print(f"Error writing to file '{json_file}': {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure you have pandas installed: pip install pandas
    SOURCE_CSV_FILE = 'data/bsa_sections.csv'
    OUTPUT_JSON_FILE = 'data/bsa_chunks.json'
    
    convert_bsa_csv_to_json(SOURCE_CSV_FILE, OUTPUT_JSON_FILE)