import os
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_documents(directory):
    """Process all PDF and image documents in the given directory"""
    texts = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if filename.endswith('.pdf'):
                with open(filepath, 'rb') as f:
                    reader = PdfReader(f)
                    text = '\n'.join([page.extract_text() for page in reader.pages])
                    texts.append({'doc_id': filename, 'text': text})
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = pytesseract.image_to_string(Image.open(filepath))
                texts.append({'doc_id': filename, 'text': text})
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for doc in texts:
        chunks = text_splitter.split_text(doc['text'])
        for i, chunk in enumerate(chunks):
            documents.append({
                'text': chunk,
                'metadata': {
                    'doc_id': doc['doc_id'],
                    'chunk_id': i,
                    'page': i+1
                }
            })
    return documents