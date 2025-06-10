from flask import Flask, request, jsonify
import requests
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Flask
app = Flask(__name__)

# Hugging Face API Configuration
HF_API_TOKEN = os.environ.get('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')  # Set di Replit Secrets
HF_SUMMARY_URL = "https://api-inference.huggingface.co/models/fransiskaarthaa/text-summarize-fix"
HF_QUESTION_URL = "https://api-inference.huggingface.co/models/meilanikizana/question-generation-indonesia"

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

def generate_questions_with_api(text, num_questions=3):
    """Gunakan Hugging Face Inference API untuk question generation"""
    if not HF_API_TOKEN:
        logger.error("HF_API_TOKEN not available")
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Bersihkan dan format input text
    cleaned_text = clean_input_text(text)
    
    payload = {
        "inputs": cleaned_text,
        "parameters": {
            "max_length": 100,
            "min_length": 10,
            "num_return_sequences": min(num_questions, 5),
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }
    
    try:
        response = requests.post(HF_QUESTION_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            questions = extract_questions_from_result(result)
            
            # Clean dan filter questions
            cleaned_questions = clean_and_filter_questions(questions)
            
            return cleaned_questions[:num_questions] if cleaned_questions else None
        
        elif response.status_code == 503:
            logger.warning("Model is loading, please try again later")
            return None
        else:
            logger.error(f"Question API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Question request error: {e}")
        return None

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

def extract_questions_from_result(result):
    """Extract questions dari berbagai format response"""
    questions = []
    
    try:
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    # Cek berbagai kemungkinan key
                    for key in ['generated_text', 'question', 'output', 'text']:
                        if key in item and item[key]:
                            questions.extend(parse_questions_from_text(item[key]))
                elif isinstance(item, str):
                    questions.extend(parse_questions_from_text(item))
        
        elif isinstance(result, dict):
            # Cek berbagai kemungkinan key
            for key in ['generated_text', 'question', 'output', 'text']:
                if key in result and result[key]:
                    if isinstance(result[key], list):
                        for q in result[key]:
                            questions.extend(parse_questions_from_text(str(q)))
                    else:
                        questions.extend(parse_questions_from_text(result[key]))
                        
        elif isinstance(result, str):
            questions.extend(parse_questions_from_text(result))
            
    except Exception as e:
        logger.error(f"Error extracting questions: {e}")
    
    return questions

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
        "message": "Pure HuggingFace Text Processing API",
        "status": "running",
        "features": ["Text Summarization", "Question Generation"],
        "models": {
            "summarization": "fransiskaarthaa/text-summarize-fix",
            "question_generation": "meilanikizana/question-generation-indonesia"
        },
        "endpoints": ["/summarize", "/generate-questions", "/process-text"],
        "note": "Pure HuggingFace API only - no manual fallback questions"
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
                "method": "HuggingFace API",
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
    """Endpoint untuk question generation saja"""
    if not HF_API_TOKEN:
        return jsonify({"error": "HF_API_TOKEN not configured"}), 500
    
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
        
        questions = generate_questions_with_api(text, num_questions)
        
        if questions and len(questions) > 0:
            return jsonify({
                "questions": questions,
                "method": "HuggingFace API",
                "question_count": len(questions),
                "text_length": len(text),
                "status": "success"
            })
        else:
            return jsonify({"error": "Failed to generate questions from HuggingFace API"}), 500
            
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
                future_questions = executor.submit(generate_questions_with_api, text, num_questions) if include_questions else None
                
                # Get results
                try:
                    summary = future_summary.result(timeout=45)  # 45 second timeout
                except Exception as e:
                    logger.error(f"Summary API failed: {e}")
                    summary = None
                
                if future_questions:
                    try:
                        questions = future_questions.result(timeout=45)  # 45 second timeout
                    except Exception as e:
                        logger.error(f"Questions API failed: {e}")
                        questions = None
        
        else:  # Sequential processing
            # Get summary first
            summary = summarize_with_api(text, max_length)
            
            # Generate questions from original text
            if include_questions:
                questions = generate_questions_with_api(text, num_questions)
        
        # Check if we got results
        if not summary:
            return jsonify({"error": "Failed to generate summary from HuggingFace API"}), 500
        
        if include_questions and (not questions or len(questions) == 0):
            return jsonify({"error": "Failed to generate questions from HuggingFace API"}), 500
        
        # Prepare response
        response = {
            "original_text": text,
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": f"{len(summary)/len(text)*100:.1f}%",
            "method": "HuggingFace API",
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
            "summarization": "fransiskaarthaa/text-summarize-fix",
            "question_generation": "meilanikizana/question-generation-indonesia"
        },
        "note": "Pure HuggingFace API implementation"
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Pure HuggingFace Text Processing API")
    print(f"📡 Port: {port}")
    print("🤖 Models:")
    print("   📝 Summarization: fransiskaarthaa/text-summarize-fix")
    print("   ❓ Questions: meilanikizana/question-generation-indonesia")
    print("🔗 Endpoints:")
    print("   GET  / - Status & Info")
    print("   POST /summarize - Text summarization only") 
    print("   POST /generate-questions - Question generation (HuggingFace only)")
    print("   POST /process-text - Dual processing (summary + questions)")
    print("   GET  /health - Health check")
    print("✨ Features:")
    print("   🎯 Pure HuggingFace API implementation")
    print("   🚫 No manual fallback questions")
    print("   🔧 Clean error handling")
    
    if not HF_API_TOKEN:
        print("❌ ERROR: HF_TOKEN not configured!")
        print("   Please set HF_TOKEN environment variable to use the API")
    else:
        print("✅ HF_TOKEN configured - Ready to use HuggingFace API")
    
    app.run(host='0.0.0.0', port=port, debug=False)
