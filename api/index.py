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

def generate_questions_with_gradio(text, num_questions=3, timeout=30):
    """Gunakan Gradio Client untuk question generation dari Hugging Face Spaces"""
    try:
        # Bersihkan dan format input text
        cleaned_text = clean_input_text(text)
        
        logger.info(f"Connecting to Gradio Space: {GRADIO_SPACE_URL}")
        logger.info(f"Input text: {cleaned_text[:100]}...")
        logger.info(f"Num questions: {num_questions}")
        
        # Initialize Gradio Client dengan timeout yang lebih pendek
        client = Client(GRADIO_SPACE_URL)
        
        # Call the predict function dengan timeout handling
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Gradio request timed out")
        
        # Set timeout alarm
        signal.alarm(timeout)
        
        try:
            result = client.predict(
                context=cleaned_text,  # parameter pertama: context (str)
                num_questions=int(num_questions),  # parameter kedua: num_questions (convert ke int, bukan float)
                api_name="/predict"
            )
        finally:
            # Cancel alarm
            signal.alarm(0)
        
        logger.info(f"Raw Gradio API Response: {result}")
        logger.info(f"Response type: {type(result)}")
        
        # Parse the result
        if result is None:
            logger.error("Gradio returned None result")
            return None
            
        if isinstance(result, str):
            # Jika result adalah string kosong atau hanya whitespace
            if not result.strip():
                logger.error("Gradio returned empty string")
                return None
                
            questions = parse_questions_from_gradio_result(result)
            logger.info(f"Parsed questions: {questions}")
        else:
            logger.error(f"Unexpected result type from Gradio: {type(result)}")
            logger.error(f"Result content: {result}")
            return None
        
        # Clean dan filter questions
        cleaned_questions = clean_and_filter_questions(questions)
        logger.info(f"Cleaned questions: {cleaned_questions}")
        
        if not cleaned_questions:
            logger.error("No valid questions after cleaning")
            return None
            
        return cleaned_questions[:num_questions] if cleaned_questions else None
        
    except TimeoutError:
        logger.error(f"Gradio request timed out after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Gradio Client error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def test_gradio_connection():
    """Test koneksi ke Gradio Spaces dengan timeout pendek"""
    try:
        logger.info("Testing Gradio connection...")
        client = Client(GRADIO_SPACE_URL)
        
        # Set timeout untuk test
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Connection test timed out")
        
        signal.alarm(15)  # 15 detik timeout untuk test
        
        try:
            # Test dengan input sederhana
            test_result = client.predict(
                context="Indonesia adalah negara kepulauan yang indah.",
                num_questions=1,
                api_name="/predict"
            )
        finally:
            signal.alarm(0)
        
        logger.info(f"Test result: {test_result}")
        logger.info(f"Test result type: {type(test_result)}")
        return test_result is not None
        
    except TimeoutError:
        logger.error("Gradio connection test timed out")
        return False
    except Exception as e:
        logger.error(f"Gradio connection test failed: {e}")
        return False

def parse_questions_from_gradio_result(result_text):
    """Parse questions dari hasil Gradio yang berupa string"""
    if not result_text or not isinstance(result_text, str):
        logger.warning(f"Invalid result_text: {result_text}")
        return []
    
    questions = []
    result_text = result_text.strip()
    
    logger.info(f"Parsing result text: {result_text}")
    
    # Method 1: Split berdasarkan newline
    lines = result_text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            # Remove numbering jika ada (1. 2. 3. dll)
            line = re.sub(r'^\d+\.?\s*', '', line)
            line = line.strip()
            
            # Pastikan berakhir dengan tanda tanya
            if line and not line.endswith('?'):
                line += '?'
            
            if line:
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
    
    # Method 3: Jika masih kosong, anggap seluruh text adalah satu pertanyaan
    if len(questions) == 0 and result_text:
        if not result_text.endswith('?'):
            result_text += '?'
        questions.append(result_text)
    
    logger.info(f"Parsed {len(questions)} questions: {questions}")
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
    if not questions:
        return []
        
    cleaned = []
    seen = set()
    
    for q in questions:
        if not q or not isinstance(q, str):
            continue
            
        # Basic cleaning
        q = q.strip()
        q = re.sub(r'\s+', ' ', q)  # Normalize whitespace
        
        # Skip if too short or too long
        if len(q) < 5 or len(q) > 200:  # Lowered minimum length
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
        
        # Less strict validation for Indonesian questions
        if not is_valid_question_relaxed(q):
            continue
        
        cleaned.append(q)
        seen.add(q_lower)
    
    return cleaned

def generate_questions_fallback(text, num_questions=3):
    """Fallback method untuk generate questions tanpa Gradio"""
    try:
        logger.info("Using fallback question generation method")
        
        # Simple rule-based question generation
        sentences = text.split('.')
        questions = []
        
        # Template pertanyaan bahasa Indonesia
        question_templates = [
            "Apa yang dimaksud dengan {}?",
            "Mengapa {} penting?",
            "Bagaimana {} dapat dijelaskan?",
            "Kapan {} terjadi?",
            "Dimana {} dapat ditemukan?"
        ]
        
        # Extract key terms (simple approach)
        import re
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        
        # Filter common words
        common_words = {'dan', 'atau', 'yang', 'untuk', 'dari', 'dengan', 'pada', 'di', 'ke', 'oleh', 'adalah', 'ini', 'itu', 'tersebut'}
        key_terms = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Generate questions
        for i, template in enumerate(question_templates[:num_questions]):
            if i < len(key_terms):
                question = template.format(key_terms[i])
                questions.append(question)
        
        # If we don't have enough questions, create generic ones
        while len(questions) < num_questions:
            if len(questions) == 0:
                questions.append("Apa informasi penting dari teks ini?")
            elif len(questions) == 1:
                questions.append("Bagaimana hal ini dapat dijelaskan lebih lanjut?")
            else:
                questions.append("Mengapa topik ini menarik untuk dibahas?")
        
        return questions[:num_questions]
        
    except Exception as e:
        logger.error(f"Fallback question generation failed: {e}")
        return None
    """Validate if the text is a proper question (relaxed version)"""
    # Must end with question mark
    if not question.endswith('?'):
        return False
    
    # Should have reasonable length
    words = question.split()
    if len(words) < 2:  # Very minimal requirement
        return False
    
    # If it contains basic question structure, accept it
    question_lower = question.lower()
    
    question_indicators = [
        'apa', 'siapa', 'dimana', 'kapan', 'mengapa', 'bagaimana', 
        'berapa', 'mana', 'kenapa', 'gimana', 'apakah', 'adakah',
        'bisakah', 'dapatkah', 'haruskah', 'akankah'
    ]
    
    # Check for question words (more lenient)
    has_question_word = any(word in question_lower for word in question_indicators)
    
    # If it has question words, accept it
    if has_question_word:
        return True
    
    # If it doesn't have obvious question words but ends with ?, 
    # and has reasonable length, accept it (could be implicit question)
    if len(words) >= 3:
        return True
    
    return False

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
        "endpoints": ["/summarize", "/generate-questions", "/process-text", "/test-gradio"],
        "note": "Using HuggingFace Inference API for summarization and Gradio Spaces for question generation"
    })

@app.route('/test-gradio', methods=['GET'])
def test_gradio():
    """Test endpoint untuk Gradio connection"""
    try:
        success = test_gradio_connection()
        return jsonify({
            "gradio_test": "success" if success else "failed",
            "space_url": GRADIO_SPACE_URL,
            "status": "connected" if success else "connection_failed"
        })
    except Exception as e:
        return jsonify({
            "gradio_test": "failed",
            "error": str(e),
            "space_url": GRADIO_SPACE_URL
        }), 500

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Endpoint untuk question generation menggunakan Gradio Spaces dengan fallback"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Field 'text' required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 10:  # Relaxed minimum length
            return jsonify({"error": "Text too short for question generation"}), 400
            
        num_questions = min(data.get('num_questions', 3), 10)  # Limit max 10
        use_fallback = data.get('use_fallback', False)  # Allow forced fallback
        
        logger.info(f"Processing request - Text length: {len(text)}, Num questions: {num_questions}")
        
        questions = None
        method_used = "unknown"
        
        # Try Gradio first (unless fallback is forced)
        if not use_fallback:
            try:
                questions = generate_questions_with_gradio(text, num_questions, timeout=25)
                method_used = "Gradio Spaces"
            except Exception as e:
                logger.warning(f"Gradio failed, trying fallback: {e}")
                questions = None
        
        # Use fallback if Gradio failed or was skipped
        if not questions or len(questions) == 0:
            logger.info("Using fallback question generation")
            questions = generate_questions_fallback(text, num_questions)
            method_used = "Fallback (Rule-based)"
        
        if questions and len(questions) > 0:
            return jsonify({
                "questions": questions,
                "method": method_used,
                "gradio_space": GRADIO_SPACE_URL if method_used == "Gradio Spaces" else None,
                "question_count": len(questions),
                "text_length": len(text),
                "status": "success",
                "fallback_used": method_used != "Gradio Spaces"
            })
        else:
            # Complete failure
            return jsonify({
                "error": "Failed to generate questions with both Gradio and fallback methods",
                "debug_info": {
                    "text_length": len(text),
                    "num_questions_requested": num_questions,
                    "gradio_space": GRADIO_SPACE_URL,
                    "methods_tried": ["Gradio Spaces", "Fallback"],
                    "suggestion": "The Gradio Space might be down or overloaded. Try again later or contact support."
                }
            }), 500
            
    except Exception as e:
        logger.error(f"Error in generate_questions: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Server error occurred",
            "details": str(e)
        }), 500

# ... (rest of the code remains the same)

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

@app.route('/health', methods=['GET'])
def health():
    """Health check untuk monitoring"""
    gradio_status = test_gradio_connection()
    
    return jsonify({
        "status": "healthy" if HF_API_TOKEN and gradio_status else "degraded",
        "hf_token_configured": bool(HF_API_TOKEN),
        "gradio_connection": "connected" if gradio_status else "failed",
        "models": {
            "summarization": "fransiskaarthaa/text-summarize-fix (HF Inference API)",
            "question_generation": f"{GRADIO_SPACE_URL} (Gradio Spaces)"
        },
        "services": {
            "huggingface_inference": "ready" if HF_API_TOKEN else "missing_token",
            "gradio_spaces": "ready" if gradio_status else "connection_failed"
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
    print("   GET  /test-gradio - Test Gradio connection")
    print("   GET  /health - Health check")
    print("✨ Features:")
    print("   🎯 HuggingFace Inference API + Gradio Spaces")
    print("   🔧 Enhanced error handling & debugging")
    print("   🧪 Gradio connection testing")
    
    if not HF_API_TOKEN:
        print("❌ WARNING: HF_API_TOKEN not configured for summarization!")
    else:
        print("✅ HF_API_TOKEN configured")
    
    print("🧪 Testing Gradio connection...")
    if test_gradio_connection():
        print("✅ Gradio Spaces connection successful")
    else:
        print("❌ WARNING: Gradio Spaces connection failed")
    
    app.run(host='0.0.0.0', port=port, debug=False)
