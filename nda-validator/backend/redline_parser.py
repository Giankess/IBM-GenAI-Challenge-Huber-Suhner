import os
import zipfile
import re
import xml.etree.ElementTree as ET
from docx import Document
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XML namespaces used in Word documents
namespaces = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
}

def extract_docx_xml(docx_path):
    """
    Extract the document.xml file from a .docx file which contains the content with tracked changes
    """
    try:
        temp_dir = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract the docx (which is a zip file)
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            # Extract only the document.xml which contains the content
            zip_ref.extract('word/document.xml', temp_dir)
        
        xml_path = os.path.join(temp_dir, 'word', 'document.xml')
        return xml_path, temp_dir
    except Exception as e:
        logger.error(f"Error extracting XML from docx: {str(e)}")
        return None, None

def cleanup_temp_dir(temp_dir):
    """
    Clean up temporary directory
    """
    try:
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {str(e)}")

def parse_redline_document(docx_path):
    """
    Parse a Word document with tracked changes (redline) and extract problematic clauses and replacements
    """
    xml_path, temp_dir = extract_docx_xml(docx_path)
    if not xml_path:
        return []
    
    try:
        # Parse the XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Register namespaces for XPath
        for prefix, uri in namespaces.items():
            ET.register_namespace(prefix, uri)
        
        # Extract paragraphs with tracked changes
        redline_data = []
        
        # Process each paragraph
        for paragraph in root.findall('.//w:p', namespaces):
            paragraph_text = ""
            deletions = []
            insertions = []
            
            # Process each run in the paragraph
            for run in paragraph.findall('.//w:r', namespaces):
                # Check if this run is deleted text
                del_nodes = run.findall('.//w:del', namespaces) + run.findall('.//w:delText', namespaces)
                if del_nodes:
                    # This is deleted text (problematic)
                    for del_node in run.findall('.//w:delText', namespaces) + run.findall('.//w:t', namespaces):
                        if del_node.text:
                            deletions.append(del_node.text)
                
                # Check if this run is inserted text
                ins_nodes = run.findall('.//w:ins', namespaces)
                if ins_nodes:
                    # This is inserted text (replacement)
                    for ins_text in run.findall('.//w:t', namespaces):
                        if ins_text.text:
                            insertions.append(ins_text.text)
                
                # Regular text
                for text in run.findall('.//w:t', namespaces):
                    if text.text:
                        paragraph_text += text.text
            
            # If we found tracked changes in this paragraph
            if deletions or insertions:
                redline_data.append({
                    "paragraph_text": paragraph_text,
                    "problematic_text": "".join(deletions),
                    "replacement_text": "".join(insertions)
                })
        
        # Clean up
        cleanup_temp_dir(temp_dir)
        
        return redline_data
    
    except Exception as e:
        logger.error(f"Error parsing redline document: {str(e)}")
        cleanup_temp_dir(temp_dir)
        return []

def parse_redline_with_python_docx(docx_path):
    """
    Alternative approach using python-docx to extract text and infer redlines
    This is a fallback method that doesn't directly access tracked changes
    but tries to identify them through formatting
    """
    try:
        doc = Document(docx_path)
        redline_data = []
        
        for para in doc.paragraphs:
            paragraph_text = para.text
            problematic_parts = []
            replacement_parts = []
            
            # Look for formatting that might indicate tracked changes
            for run in para.runs:
                if run.font.strike:  # Strikethrough text is often used for deletions
                    problematic_parts.append(run.text)
                elif run.font.color.rgb and run.font.color.rgb != (0, 0, 0):  # Colored text often indicates additions
                    replacement_parts.append(run.text)
            
            if problematic_parts or replacement_parts:
                redline_data.append({
                    "paragraph_text": paragraph_text,
                    "problematic_text": "".join(problematic_parts),
                    "replacement_text": "".join(replacement_parts)
                })
        
        return redline_data
    
    except Exception as e:
        logger.error(f"Error parsing with python-docx: {str(e)}")
        return []

def convert_redline_to_training_data(redline_data):
    """
    Convert extracted redline data to training data format
    """
    training_data = {
        "clauses": []
    }
    
    for item in redline_data:
        if item["problematic_text"]:
            training_data["clauses"].append({
                "text": item["problematic_text"],
                "is_problematic": True,
                "replacement": item["replacement_text"] if item["replacement_text"] else None
            })
        
        # Also add the paragraph text as context
        if item["paragraph_text"] and item["paragraph_text"] != item["problematic_text"]:
            training_data["clauses"].append({
                "text": item["paragraph_text"],
                "is_problematic": False
            })
    
    return training_data

def process_redline_document(docx_path, output_path=None):
    """
    Process a redline document and extract training data
    """
    # Try the XML method first
    redline_data = parse_redline_document(docx_path)
    
    # If that didn't work, try the python-docx method
    if not redline_data:
        redline_data = parse_redline_with_python_docx(docx_path)
    
    # Convert to training data format
    training_data = convert_redline_to_training_data(redline_data)
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
    
    return training_data

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        docx_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Processing redline document: {docx_path}")
        training_data = process_redline_document(docx_path, output_path)
        
        print(f"Extracted {len(training_data['clauses'])} clauses")
        problematic_count = sum(1 for clause in training_data['clauses'] if clause['is_problematic'])
        print(f"Found {problematic_count} problematic clauses with replacements")
    else:
        print("Usage: python redline_parser.py <docx_path> [output_path]")
