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
        "Authorization": f"Bearer {HF_API_TOKEN}" if HF_API_TOKEN else None
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
        "Authorization": f"Bearer {HF_API_TOKEN}" if HF_API_TOKEN else None
    }
    
    # Format input sesuai dengan model question generation
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 200,
            "num_return_sequences": num_questions,
            "do_sample": True,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(HF_QUESTION_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            questions = []
            
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and 'generated_text' in item:
                        questions.append(item['generated_text'].strip())
                    elif isinstance(item, str):
                        questions.append(item.strip())
            elif isinstance(result, dict) and 'generated_text' in result:
                questions = [result['generated_text'].strip()]
            elif isinstance(result, str):
                # Jika response berupa string, split berdasarkan pattern
                questions = parse_questions_from_text(result)
            
            # Clean up questions - remove duplicates dan format
            questions = clean_questions(questions)
            return questions[:num_questions]  # Limit sesuai request
        else:
            logger.error(f"Question API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Question request error: {e}")
        return None

def parse_questions_from_text(text):
    """Parse questions dari text response"""
    # Split berdasarkan pattern question (ends with ?)
    questions = re.split(r'[.!](?=\s*[A-Z])', text)
    
    # Filter hanya yang berupa pertanyaan (ends with ?)
    questions = [q.strip() for q in questions if q.strip().endswith('?')]
    
    return questions

def clean_questions(questions):
    """Bersihkan dan format questions"""
    cleaned = []
    seen = set()
    
    for q in questions:
        q = q.strip()
        if q and q.endswith('?') and q not in seen and len(q) > 10:
            # Capitalize first letter
            q = q[0].upper() + q[1:] if len(q) > 1 else q.upper()
            cleaned.append(q)
            seen.add(q)
    
    return cleaned

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

def simple_question_generation(text, num_questions=3):
    """Fallback: simple question generation based on text analysis"""
    sentences = text.split('. ')
    questions = []
    
    # Generate questions based on common patterns
    question_templates = [
        "Apa yang dimaksud dengan {}?",
        "Bagaimana cara {}?",
        "Mengapa {}?",
        "Kapan {}?",
        "Di mana {}?"
    ]
    
    # Extract key phrases (simplified)
    words = text.lower().split()
    common_words = {'adalah', 'dengan', 'yang', 'untuk', 'dalam', 'pada', 'akan', 'dapat', 'atau', 'dan', 'ini', 'itu'}
    key_words = [w for w in words if len(w) > 3 and w not in common_words]
    
    # Generate simple questions
    if key_words:
        for i, template in enumerate(question_templates[:num_questions]):
            if i < len(key_words):
                questions.append(template.format(key_words[i]))
    
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
        "storage_usage": "Minimal - no local models"
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Endpoint untuk summarization - disesuaikan dengan ai.ts"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Field 'text' required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 50:
            return jsonify({"error": "Text too short for summarization"}), 400
            
        # Sesuai dengan format request dari ai.ts
        length_mapping = {
            "short": 100,
            "medium": 150,
            "long": 200
        }
        
        length = data.get('length', 'medium')
        max_length = length_mapping.get(length, 150)
        include_questions = data.get('include_questions', False)  # Default false untuk backward compatibility
        num_questions = data.get('num_questions', 3)
        use_api = data.get('use_api', True)
        
        # Variables untuk hasil
        summary = None
        questions = []
        methods_used = {}
        
        if include_questions:
            # Jika diminta questions, jalankan parallel processing
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                future_summary = executor.submit(summarize_with_api, text, max_length) if use_api else None
                future_questions = executor.submit(generate_questions_with_api, text, num_questions) if use_api else None
                
                # Get results
                if future_summary:
                    try:
                        summary = future_summary.result(timeout=30)
                        methods_used['summary'] = "HuggingFace API"
                    except Exception as e:
                        logger.error(f"Summary API failed: {e}")
                        summary = None
                
                if future_questions:
                    try:
                        questions = future_questions.result(timeout=30)
                        methods_used['questions'] = "HuggingFace API"
                    except Exception as e:
                        logger.error(f"Questions API failed: {e}")
                        questions = None
            
            # Fallbacks
            if not summary:
                summary = simple_summarize(text)
                methods_used['summary'] = "Simple fallback"
            
            if not questions:
                questions = simple_question_generation(text, num_questions)
                methods_used['questions'] = "Simple fallback"
        
        else:
            # Hanya summarization
            if use_api:
                summary = summarize_with_api(text, max_length)
                methods_used['summary'] = "HuggingFace API"
            
            if not summary:
                summary = simple_summarize(text)
                methods_used['summary'] = "Simple fallback"
        
        if summary:
            # Response format yang sesuai dengan ai.ts expectation
            response = {
                "summary": summary,  # Format yang diharapkan ai.ts
                "method": methods_used.get('summary', 'Unknown'),
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": f"{len(summary)/len(text)*100:.1f}%",
                "status": "success"
            }
            
            # Tambahkan questions jika diminta
            if include_questions:
                response["questions"] = questions
                response["question_count"] = len(questions)
                response["question_method"] = methods_used.get('questions', 'Unknown')
            
            return jsonify(response)
        else:
            return jsonify({"error": "Failed to generate summary"}), 500
            
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
        if use_api:
            questions = generate_questions_with_api(text, num_questions)
            method_used = "HuggingFace API"
        
        # Fallback to simple method
        if not questions:
            questions = simple_question_generation(text, num_questions)
            method_used = "Simple fallback"
        
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
        
        if processing_mode == 'parallel':
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
                        methods_used['questions'] = "HuggingFace API"
                    except Exception as e:
                        logger.error(f"Questions API failed: {e}")
                        questions = None
        
        else:  # Sequential processing
            # Get summary first
            if use_api:
                summary = summarize_with_api(text, max_length)
                methods_used['summary'] = "HuggingFace API" if summary else None
            
            # Then generate questions (could be based on summary or original text)
            if include_questions and use_api:
                input_for_questions = summary if summary else text  # Use summary if available
                questions = generate_questions_with_api(input_for_questions, num_questions)
                methods_used['questions'] = "HuggingFace API" if questions else None
        
        # Fallbacks
        if not summary:
            summary = simple_summarize(text)
            methods_used['summary'] = "Simple fallback"
        
        if include_questions and not questions:
            questions = simple_question_generation(text, num_questions)
            methods_used['questions'] = "Simple fallback"
        
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
        "storage": "minimal",
        "models_available": {
            "summarization": bool(HF_API_TOKEN),
            "question_generation": bool(HF_API_TOKEN)
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Enhanced Text Processing API")
    print(f"📡 Port: {port}")
    print("💾 Storage usage: MINIMAL (no local models)")
    print("🤖 Models:")
    print("   📝 Summarization: fransiskaarthaa/text-summarize")
    print("   ❓ Questions: meilanikizana/question-generation-indonesia")
    print("🔗 Endpoints:")
    print("   GET  / - Status & Info")
    print("   POST /summarize - Text summarization only") 
    print("   POST /generate-questions - Question generation only")
    print("   POST /process-text - Dual processing (summary + questions)")
    print("   GET  /health - Health check")
    
    if not HF_API_TOKEN:
        print("⚠️  Warning: No HF_TOKEN set. API methods will use fallbacks.")
        print("   Add HF_TOKEN in environment variables for full functionality")
    else:
        print("✅ HF_TOKEN configured - API methods available")
    
    app.run(host='0.0.0.0', port=port, debug=False)
