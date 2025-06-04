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
HF_API_TOKEN = os.environ.get('HF_TOKEN')  # Set di Replit Secrets
HF_SUMMARY_URL = "https://api-inference.huggingface.co/models/fransiskaarthaa/text-summarize"
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
        response = requests.post(HF_SUMMARY_URL, headers=headers, json=payload)
        
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
            "num_return_sequences": min(num_questions, 5),  # Limit untuk menghindari timeout
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
            
            # Jika tidak mendapat cukup pertanyaan, coba dengan input yang lebih pendek
            if len(cleaned_questions) < num_questions and len(cleaned_text) > 200:
                # Coba dengan text yang dipotong
                short_text = cleaned_text[:200] + "..."
                return generate_questions_with_api(short_text, num_questions)
            
            return cleaned_questions[:num_questions]
        
        elif response.status_code == 503:
            logger.warning("Model is loading, will retry...")
            # Model sedang loading, bisa dicoba lagi setelah beberapa detik
            import time
            time.sleep(5)
            # Recursive call dengan fallback ke simple method jika masih gagal
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

def fallback_question_generation(text, num_questions=3):
    """Improved fallback question generation berdasarkan analisis teks"""
    sentences = text.split('. ')
    questions = []
    
    # Analisis kata kunci dari teks
    words = text.lower().split()
    stop_words = {
        'adalah', 'dengan', 'yang', 'untuk', 'dalam', 'pada', 'akan', 'dapat', 
        'atau', 'dan', 'ini', 'itu', 'dari', 'ke', 'di', 'oleh', 'karena',
        'sehingga', 'tetapi', 'namun', 'juga', 'jika', 'bila', 'ketika'
    }
    
    # Extract key entities and concepts
    key_phrases = []
    for sentence in sentences[:3]:  # Fokus pada 3 kalimat pertama
        sentence_words = sentence.lower().split()
        meaningful_words = [w for w in sentence_words if len(w) > 3 and w not in stop_words]
        if meaningful_words:
            key_phrases.extend(meaningful_words[:2])  # Ambil 2 kata penting per kalimat
    
    # Generate contextual questions
    if key_phrases:
        # Pertanyaan berdasarkan konsep utama
        if len(key_phrases) > 0:
            questions.append(f"Apa yang dimaksud dengan {key_phrases[0]} dalam konteks ini?")
        
        if len(key_phrases) > 1:
            questions.append(f"Bagaimana hubungan antara {key_phrases[0]} dan {key_phrases[1]}?")
        
        if len(sentences) > 1:
            questions.append("Apa kesimpulan utama yang dapat diambil dari teks ini?")
    
    # Fallback questions jika tidak ada key phrases
    if not questions:
        questions = [
            "Apa tema utama yang dibahas dalam teks ini?",
            "Bagaimana penjelasan dari topik yang dibahas?",
            "Apa yang dapat dipelajari dari informasi ini?"
        ]
    
    return questions[:num_questions]

@app.route('/', methods=['GET'])
def home():
    """Status check"""
    return jsonify({
        "message": "Enhanced Text Processing API",
        "status": "running",
        "features": ["Text Summarization", "Question Generation", "Dual Model Processing"],
        "models": {
            "summarization": "fransiskaarthaa/text-summarize",
            "question_generation": "meilanikizana/question-generation-indonesia"
        },
        "endpoints": ["/summarize", "/generate-questions", "/process-text"],
        "improvements": [
            "Direct model-based question generation",
            "Intelligent question parsing and validation",
            "Improved fallback mechanisms",
            "Better text preprocessing"
        ]
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Endpoint untuk summarization saja"""
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
        use_api = data.get('use_api', True)
        
        summary = None
        method_used = ""
        
        # Try HuggingFace API first
        if use_api and HF_API_TOKEN:
            summary = summarize_with_api(text, max_length)
            method_used = "HuggingFace API"
        
        # Fallback to simple method
        if not summary:
            summary = simple_summarize(text)
            method_used = "Simple fallback"
        
        if summary:
            return jsonify({
                "summary": summary,
                "method": method_used,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": f"{len(summary)/len(text)*100:.1f}%",
                "status": "success"
            })
        else:
            return jsonify({"error": "Failed to summarize"}), 500
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Server error occurred"}), 500

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Endpoint untuk question generation saja"""
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
        use_api = data.get('use_api', True)
        
        questions = None
        method_used = ""
        
        # Try HuggingFace API first
        if use_api and HF_API_TOKEN:
            questions = generate_questions_with_api(text, num_questions)
            method_used = "HuggingFace API (Direct Model Generation)"
        
        # Fallback to improved simple method
        if not questions or len(questions) == 0:
            questions = fallback_question_generation(text, num_questions)
            method_used = "Improved contextual fallback"
        
        if questions:
            return jsonify({
                "questions": questions,
                "method": method_used,
                "question_count": len(questions),
                "text_length": len(text),
                "status": "success"
            })
        else:
            return jsonify({"error": "Failed to generate questions"}), 500
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Server error occurred"}), 500

@app.route('/process-text', methods=['POST'])
def process_text():
    """Endpoint untuk dual processing (summary + questions)"""
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
        use_api = data.get('use_api', True)
        include_questions = data.get('include_questions', True)
        processing_mode = data.get('mode', 'parallel')  # 'parallel' or 'sequential'
        
        summary = None
        questions = []
        methods_used = {}
        
        if processing_mode == 'parallel' and HF_API_TOKEN:
            # Parallel processing using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                future_summary = executor.submit(summarize_with_api, text, max_length) if use_api else None
                future_questions = executor.submit(generate_questions_with_api, text, num_questions) if (use_api and include_questions) else None
                
                # Get results
                if future_summary:
                    try:
                        summary = future_summary.result(timeout=30)  # 30 second timeout
                        methods_used['summary'] = "HuggingFace API"
                    except Exception as e:
                        logger.error(f"Summary API failed: {e}")
                        summary = None
                
                if future_questions:
                    try:
                        questions = future_questions.result(timeout=30)  # 30 second timeout
                        methods_used['questions'] = "HuggingFace API (Direct Model)"
                    except Exception as e:
                        logger.error(f"Questions API failed: {e}")
                        questions = None
        
        else:  # Sequential processing
            # Get summary first
            if use_api and HF_API_TOKEN:
                summary = summarize_with_api(text, max_length)
                methods_used['summary'] = "HuggingFace API" if summary else None
            
            # Generate questions from original text (not summary)
            if include_questions and use_api and HF_API_TOKEN:
                questions = generate_questions_with_api(text, num_questions)
                methods_used['questions'] = "HuggingFace API (Direct Model)" if questions else None
        
        # Fallbacks
        if not summary:
            summary = simple_summarize(text)
            methods_used['summary'] = "Simple fallback"
        
        if include_questions and (not questions or len(questions) == 0):
            questions = fallback_question_generation(text, num_questions)
            methods_used['questions'] = "Improved contextual fallback"
        
        # Prepare response
        response = {
            "original_text": text,
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary) if summary else 0,
            "compression_ratio": f"{len(summary)/len(text)*100:.1f}%" if summary else "0%",
            "methods_used": methods_used,
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
        "status": "healthy", 
        "models_available": {
            "summarization": bool(HF_API_TOKEN),
            "question_generation": bool(HF_API_TOKEN)
        },
        "improvements": [
            "Direct model-based question generation",
            "Intelligent question validation",
            "Enhanced text preprocessing",
            "Better error handling"
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Enhanced Text Processing API v2.0")
    print(f"📡 Port: {port}")
    print("🤖 Models:")
    print("   📝 Summarization: fransiskaarthaa/text-summarize")
    print("   ❓ Questions: meilanikizana/question-generation-indonesia (Direct Generation)")
    print("🔗 Endpoints:")
    print("   GET  / - Status & Info")
    print("   POST /summarize - Text summarization only") 
    print("   POST /generate-questions - Question generation (Direct from model)")
    print("   POST /process-text - Dual processing (summary + questions)")
    print("   GET  /health - Health check")
    print("✨ Improvements:")
    print("   🎯 Direct model-based question generation")
    print("   🔍 Intelligent question parsing and validation")
    print("   🧹 Better text preprocessing")
    print("   🔧 Enhanced error handling")
    
    if not HF_API_TOKEN:
        print("⚠️  Warning: No HF_TOKEN set. Using improved fallback methods.")
        print("   Add HF_TOKEN in environment variables for full functionality")
    else:
        print("✅ HF_TOKEN configured - Direct model generation available")
    
    app.run(host='0.0.0.0', port=port, debug=False)
