import pandas as pd
import json
import re

def clean_text(text):
    """Clean and normalize text content"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text

def format_section_number(section):
    """Ensure consistent section number formatting"""
    section = str(section).strip()
    # Remove non-numeric characters from section number
    section = re.sub(r'[^\d]', '', section)
    return section.zfill(3)  # Pad with zeros for consistent sorting

def build_chunks(input_csv, output_json):
    """Process CSV and create structured chunks"""
    chunks = []
    
    df = pd.read_csv(input_csv, encoding="utf-8")
    
    for i, row in df.iterrows():
        section = format_section_number(row.get("Section", ""))
        heading = clean_text(row.get("Section _name", ""))
        content = clean_text(row.get("Description", ""))
        chapter = clean_text(row.get("Chapter_name", ""))
        
        # Create structured chunk with multiple representations
        chunk_text = (
            f"BNS Section {section}: {heading}\n"
            f"Chapter: {chapter}\n\n"
            f"Legal Text:\n{content}\n"
            f"(End of Section {section})"
        )
        
        chunks.append({
            "id": f"bns_{section}_{i}",
            "section": section,
            "heading": heading,
            "content": chunk_text,
            "raw_content": content,  # Keep original for reference
            "chapter": chapter,
            "section_display": f"Section {section}",  # For display purposes
        })
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Built {len(chunks)} chunks.")

input_file = "data/bns_sections.csv"
output_file = "data/bns_chunks.json"
build_chunks(input_file, output_file)
