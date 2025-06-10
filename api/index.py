from flask import Flask, request, jsonify
import requests
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from gradio_client import Client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Flask
app = Flask(__name__)

# Hugging Face API Configuration
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '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')
HF_SUMMARY_URL = "https://api-inference.huggingface.co/models/fransiskaarthaa/text-summarize-fix"

# Gradio Spaces Configuration
GRADIO_SPACE_URL = "meilanikizana/indonesian-question-generator"

def summarize_with_api(text, max_length=150):
    """Gunakan Hugging Face Inference API untuk summarization"""
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": 40,
            "do_sample": False
        }
    }
    
    try:
        response = requests.post(HF_SUMMARY_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('summary_text', '')
            return result.get('summary_text', '')
        else:
            logger.error(f"Summary API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Summary request error: {e}")
        return None

import requests

def generate_questions_with_gradio(text, num_questions=3):
    try:
        cleaned_text = clean_input_text(text)
        logger.info(f"Sending request to Gradio Space with context: {cleaned_text}")

        payload = {
            "data": [cleaned_text, float(num_questions)]
        }

        response = requests.post(
            "https://meilanikizana-indonesian-question-generator.hf.space/run/predict",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            # Result structure: {'data': ['string hasil']}
            raw_text = result.get('data', [None])[0]

            if isinstance(raw_text, str) and raw_text.strip():
                logger.info(f"Gradio raw response: {raw_text}")
                questions = parse_questions_from_gradio_result(raw_text)
                return clean_and_filter_questions(questions)
            else:
                logger.error("Gradio result kosong atau bukan string")
                return None
        else:
            logger.error(f"Gradio HTTP Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        logger.exception("Gagal call Gradio via requests:")
        return None


def parse_questions_from_gradio_result(result_text):
    """Parse questions dari hasil Gradio yang berupa string"""
    if not result_text or not isinstance(result_text, str):
        return []
    
    questions = []
    
    # Method 1: Split berdasarkan newline (biasanya Gradio return dengan newline)
    lines = result_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            # Remove numbering jika ada (1. 2. 3. dll)
            line = re.sub(r'^\d+\.?\s*', '', line)
            
            # Pastikan berakhir dengan tanda tanya
            if not line.endswith('?'):
                line += '?'
            
            questions.append(line)
    
    # Method 2: Jika tidak ada newline, coba split dengan delimiter lain
    if len(questions) == 0:
        # Coba split berdasarkan pola angka (1. 2. 3.)
        parts = re.split(r'\d+\.?\s*', result_text)
        for part in parts:
            part = part.strip()
            if part and len(part) > 5:
                if not part.endswith('?'):
                    part += '?'
                questions.append(part)
    
    # Method 3: Fallback ke method parsing original
    if len(questions) == 0:
        questions = parse_questions_from_text(result_text)
    
    return questions

def clean_input_text(text):
    """Bersihkan input text untuk model"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Batasi panjang text untuk menghindari timeout
    if len(text) > 500:
        # Ambil 500 karakter pertama dan pastikan berakhir di kalimat
        truncated = text[:500]
        last_period = truncated.rfind('.')
        if last_period > 200:  # Pastikan masih ada konten yang cukup
            text = truncated[:last_period + 1]
        else:
            text = truncated + "."
    
    return text

def parse_questions_from_text(text):
    """Parse questions dari text dengan berbagai delimiter"""
    if not text or not isinstance(text, str):
        return []
    
    questions = []
    
    # Method 1: Split by question mark
    potential_questions = re.split(r'\?+', text)
    for q in potential_questions:
        q = q.strip()
        if q and len(q) > 5:  # Filter yang terlalu pendek
            # Tambahkan tanda tanya kembali
            q = q + '?'
            questions.append(q)
    
    # Method 2: Find sentences ending with question mark
    question_pattern = r'[^.!?]*\?+'
    found_questions = re.findall(question_pattern, text)
    for q in found_questions:
        q = q.strip()
        if q and len(q) > 5:
            questions.append(q)
    
    # Method 3: Split by common delimiters and check for question words
    delimiters = ['\n', '|', ';', '.', '!']
    for delimiter in delimiters:
        if delimiter in text:
            parts = text.split(delimiter)
            for part in parts:
                part = part.strip()
                if part and (part.endswith('?') or contains_question_words(part)):
                    if not part.endswith('?'):
                        part += '?'
                    questions.append(part)
    
    return questions

def contains_question_words(text):
    """Check if text contains Indonesian question words"""
    question_words = [
        'apa', 'siapa', 'dimana', 'kapan', 'mengapa', 'bagaimana', 
        'berapa', 'mana', 'kenapa', 'gimana', 'apakah', 'adakah'
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in question_words)

def clean_and_filter_questions(questions):
    """Bersihkan dan filter questions yang berkualitas"""
    cleaned = []
    seen = set()
    
    for q in questions:
        if not q or not isinstance(q, str):
            continue
            
        # Basic cleaning
        q = q.strip()
        q = re.sub(r'\s+', ' ', q)  # Normalize whitespace
        
        # Skip if too short or too long
        if len(q) < 10 or len(q) > 200:
            continue
        
        # Ensure it ends with question mark
        if not q.endswith('?'):
            q += '?'
        
        # Capitalize first letter
        if q:
            q = q[0].upper() + q[1:] if len(q) > 1 else q.upper()
        
        # Skip duplicates (case insensitive)
        q_lower = q.lower()
        if q_lower in seen:
            continue
        
        # Skip if doesn't look like a proper question
        if not is_valid_question(q):
            continue
        
        cleaned.append(q)
        seen.add(q_lower)
    
    return cleaned

def is_valid_question(question):
    """Validate if the text is a proper question"""
    # Must end with question mark
    if not question.endswith('?'):
        return False
    
    # Should contain at least one question word or be interrogative
    question_lower = question.lower()
    
    question_indicators = [
        'apa', 'siapa', 'dimana', 'kapan', 'mengapa', 'bagaimana', 
        'berapa', 'mana', 'kenapa', 'gimana', 'apakah', 'adakah',
        'bisakah', 'dapatkah', 'haruskah', 'akankah'
    ]
    
    # Check for question words
    has_question_word = any(word in question_lower for word in question_indicators)
    
    # Check for interrogative structure (starts with question word)
    starts_with_question = any(question_lower.startswith(word) for word in question_indicators)
    
    # Check if it's not just a statement with question mark
    if not has_question_word and not starts_with_question:
        return False
    
    # Should have reasonable length and structure
    words = question.split()
    if len(words) < 3:  # Too short
        return False
    
    return True

def simple_summarize(text, max_sentences=3):
    """Fallback: simple sentence-based summarization"""
    sentences = text.split('. ')
    if len(sentences) <= max_sentences:
        return text
    
    # Ambil kalimat pertama, tengah, dan akhir
    selected = []
    if len(sentences) > 0:
        selected.append(sentences[0])  # Kalimat pertama
    if len(sentences) > 2:
        mid = len(sentences) // 2
        selected.append(sentences[mid])  # Kalimat tengah
    if len(sentences) > 1:
        selected.append(sentences[-1] if sentences[-1] else sentences[-2])  # Kalimat akhir
    
    return '. '.join(selected) + '.'

@app.route('/', methods=['GET'])
def home():
    """Status check"""
    return jsonify({
        "message": "HuggingFace + Gradio Spaces Text Processing API",
        "status": "running",
        "features": ["Text Summarization", "Question Generation"],
        "models": {
            "summarization": "fransiskaarthaa/text-summarize-fix (HF Inference API)",
            "question_generation": "meilanikizana/indonesian-question-generator (Gradio Spaces)"
        },
        "endpoints": ["/summarize", "/generate-questions", "/process-text"],
        "note": "Using HuggingFace Inference API for summarization and Gradio Spaces for question generation"
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Endpoint untuk summarization saja"""
    if not HF_API_TOKEN:
        return jsonify({"error": "HF_API_TOKEN not configured"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Field 'text' required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 50:
            return jsonify({"error": "Text too short for summarization"}), 400
            
        # Determine max_length based on requested length
        length_mapping = {
            "short": 100,
            "medium": 150,
            "long": 200
        }
        
        length = data.get('length', 'medium')
        max_length = length_mapping.get(length, 150)
        
        summary = summarize_with_api(text, max_length)
        
        if summary:
            return jsonify({
                "summary": summary,
                "method": "HuggingFace Inference API",
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": f"{len(summary)/len(text)*100:.1f}%",
                "status": "success"
            })
        else:
            return jsonify({"error": "Failed to generate summary from HuggingFace API"}), 500
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Server error occurred"}), 500

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Endpoint untuk question generation menggunakan Gradio Spaces"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Field 'text' required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 30:
            return jsonify({"error": "Text too short for question generation"}), 400
            
        num_questions = min(data.get('num_questions', 3), 10)  # Limit max 10
        
        questions = generate_questions_with_gradio(text, num_questions)
        
        if questions and len(questions) > 0:
            return jsonify({
                "questions": questions,
                "method": "Gradio Spaces",
                "gradio_space": GRADIO_SPACE_URL,
                "question_count": len(questions),
                "text_length": len(text),
                "status": "success"
            })
        else:
            return jsonify({"error": "Failed to generate questions from Gradio Spaces"}), 500
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Server error occurred"}), 500

@app.route('/process-text', methods=['POST'])
def process_text():
    """Endpoint untuk dual processing (summary + questions)"""
    if not HF_API_TOKEN:
        return jsonify({"error": "HF_API_TOKEN not configured"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Field 'text' required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 50:
            return jsonify({"error": "Text too short for processing"}), 400
            
        # Parameters
        length_mapping = {
            "short": 100,
            "medium": 150,
            "long": 200
        }
        
        length = data.get('length', 'medium')
        max_length = length_mapping.get(length, 150)
        num_questions = min(data.get('num_questions', 3), 10)
        include_questions = data.get('include_questions', True)
        processing_mode = data.get('mode', 'parallel')  # 'parallel' or 'sequential'
        
        summary = None
        questions = []
        
        if processing_mode == 'parallel':
            # Parallel processing using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                future_summary = executor.submit(summarize_with_api, text, max_length)
                future_questions = executor.submit(generate_questions_with_gradio, text, num_questions) if include_questions else None
                
                # Get results
                try:
                    summary = future_summary.result(timeout=60)  # 60 second timeout for Gradio
                except Exception as e:
                    logger.error(f"Summary API failed: {e}")
                    summary = None
                
                if future_questions:
                    try:
                        questions = future_questions.result(timeout=60)  # 60 second timeout for Gradio
                    except Exception as e:
                        logger.error(f"Questions API failed: {e}")
                        questions = None
        
        else:  # Sequential processing
            # Get summary first
            summary = summarize_with_api(text, max_length)
            
            # Generate questions from original text
            if include_questions:
                questions = generate_questions_with_gradio(text, num_questions)
        
        # Check if we got results
        if not summary:
            return jsonify({"error": "Failed to generate summary from HuggingFace API"}), 500
        
        if include_questions and (not questions or len(questions) == 0):
            return jsonify({"error": "Failed to generate questions from Gradio Spaces"}), 500
        
        # Prepare response
        response = {
            "original_text": text,
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": f"{len(summary)/len(text)*100:.1f}%",
            "method": {
                "summarization": "HuggingFace Inference API",
                "question_generation": "Gradio Spaces"
            },
            "processing_mode": processing_mode,
            "status": "success"
        }
        
        if include_questions:
            response.update({
                "questions": questions,
                "question_count": len(questions) if questions else 0
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in process_text: {e}")
        return jsonify({"error": "Server error occurred"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check untuk monitoring"""
    return jsonify({
        "status": "healthy" if HF_API_TOKEN else "missing_token",
        "hf_token_configured": bool(HF_API_TOKEN),
        "models": {
            "summarization": "fransiskaarthaa/text-summarize-fix (HF Inference API)",
            "question_generation": f"{GRADIO_SPACE_URL} (Gradio Spaces)"
        },
        "services": {
            "huggingface_inference": "ready" if HF_API_TOKEN else "missing_token",
            "gradio_spaces": "ready"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting HuggingFace + Gradio Spaces Text Processing API")
    print(f"📡 Port: {port}")
    print("🤖 Models:")
    print("   📝 Summarization: fransiskaarthaa/text-summarize-fix (HF Inference API)")
    print(f"   ❓ Questions: {GRADIO_SPACE_URL} (Gradio Spaces)")
    print("🔗 Endpoints:")
    print("   GET  / - Status & Info")
    print("   POST /summarize - Text summarization (HF Inference API)") 
    print("   POST /generate-questions - Question generation (Gradio Spaces)")
    print("   POST /process-text - Dual processing (summary + questions)")
    print("   GET  /health - Health check")
    print("✨ Features:")
    print("   🎯 HuggingFace Inference API + Gradio Spaces")
    print("   🔧 Clean error handling")
    print("   ⚡ Parallel processing support")
    
    if not HF_API_TOKEN:
        print("❌ WARNING: HF_API_TOKEN not configured for summarization!")
        print("   Summarization will not work without HF token")
    else:
        print("✅ HF_API_TOKEN configured - Ready to use HuggingFace Inference API")
    
    print("✅ Gradio Spaces configured - Ready to use question generation")
    
    app.run(host='0.0.0.0', port=port, debug=False)
