from flask import Flask, request, jsonify
import requests
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Flask
app = Flask(__name__)

# Hugging Face API Configuration
HF_API_TOKEN = os.environ.get('HF_TOKEN')
HF_SUMMARY_URL = "fransiskaarthaa/text-summarize"
HF_QUESTION_URL = "meilanikizana/indonesia-question-generation-model"

def summarize_with_api(text, max_length=150):
    """Gunakan Hugging Face Inference API untuk summarization"""
    if not HF_API_TOKEN:
        logger.warning("No HF_TOKEN available for summarization")
        return None
        
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
        elif response.status_code == 503:
            # Model loading, retry once after delay
            logger.info("Model loading, retrying in 10 seconds...")
            time.sleep(10)
            response = requests.post(HF_SUMMARY_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('summary_text', '')
                return result.get('summary_text', '')
        
        logger.error(f"Summary API Error: {response.status_code} - {response.text}")
        return None
            
    except Exception as e:
        logger.error(f"Summary request error: {e}")
        return None

def generate_questions_with_api(text, num_questions=3):
    """Gunakan Hugging Face Inference API untuk question generation"""
    if not HF_API_TOKEN:
        logger.warning("No HF_TOKEN available for question generation")
        return None
        
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 100,
            "num_return_sequences": num_questions,
            "do_sample": True,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(HF_QUESTION_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            questions = []
            
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and 'generated_text' in item:
                        q = item['generated_text'].strip()
                        if q and q.endswith('?') and len(q) > 10:
                            questions.append(q)
                    elif isinstance(item, str):
                        q = item.strip()
                        if q and q.endswith('?') and len(q) > 10:
                            questions.append(q)
            elif isinstance(result, dict) and 'generated_text' in result:
                q = result['generated_text'].strip()
                if q and q.endswith('?') and len(q) > 10:
                    questions = [q]
            
            # Remove duplicates and limit
            seen = set()
            unique_questions = []
            for q in questions:
                if q not in seen:
                    unique_questions.append(q)
                    seen.add(q)
            
            return unique_questions[:num_questions]
            
        elif response.status_code == 503:
            # Model loading, retry once after delay
            logger.info("Question model loading, retrying in 10 seconds...")
            time.sleep(10)
            response = requests.post(HF_QUESTION_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                questions = []
                
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and 'generated_text' in item:
                            q = item['generated_text'].strip()
                            if q and q.endswith('?') and len(q) > 10:
                                questions.append(q)
                
                # Remove duplicates and limit
                seen = set()
                unique_questions = []
                for q in questions:
                    if q not in seen:
                        unique_questions.append(q)
                        seen.add(q)
                
                return unique_questions[:num_questions]
        
        logger.error(f"Question API Error: {response.status_code} - {response.text}")
        return None
            
    except Exception as e:
        logger.error(f"Question request error: {e}")
        return None

def simple_fallback_summary(text, max_sentences=3):
    """Simple fallback summary - only used when API completely fails"""
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    if len(sentences) <= max_sentences:
        return text
    
    # Take first, middle, and last sentences
    selected = []
    if sentences:
        selected.append(sentences[0])
    if len(sentences) > 2:
        selected.append(sentences[len(sentences)//2])
    if len(sentences) > 1:
        selected.append(sentences[-1])
    
    return ' '.join(selected)

@app.route('/', methods=['GET'])
def home():
    """Status check"""
    return jsonify({
        "message": "Enhanced Text Processing API",
        "status": "running",
        "features": ["Dual Model Processing", "Text Summarization", "Question Generation"],
        "models": {
            "summarization": "fransiskaarthaa/text-summarize",
            "question_generation": "meilanikizana/question-generation-indonesia"
        },
        "endpoints": ["/summarize"],
        "api_status": "ready" if HF_API_TOKEN else "limited (no token)"
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Main endpoint - handles both summarization and question generation"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Field 'text' required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 50:
            return jsonify({"error": "Text too short for processing (minimum 50 characters)"}), 400
            
        # Parameters
        length_mapping = {
            "short": 100,
            "medium": 150,
            "long": 200
        }
        
        length = data.get('length', 'medium')
        max_length = length_mapping.get(length, 150)
        include_questions = data.get('include_questions', True)
        num_questions = min(data.get('num_questions', 3), 5)  # Limit to 5 max
        
        summary = None
        questions = []
        processing_method = {}
        
        if include_questions:
            # Parallel processing for both summary and questions
            logger.info("Starting parallel processing...")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks simultaneously
                future_summary = executor.submit(summarize_with_api, text, max_length)
                future_questions = executor.submit(generate_questions_with_api, text, num_questions)
                
                # Get results with timeout
                try:
                    summary = future_summary.result(timeout=45)
                    processing_method['summary'] = "HuggingFace API" if summary else "Failed"
                except Exception as e:
                    logger.error(f"Summary task failed: {e}")
                    processing_method['summary'] = "Failed"
                
                try:
                    questions = future_questions.result(timeout=45)
                    processing_method['questions'] = "HuggingFace API" if questions else "Failed"
                except Exception as e:
                    logger.error(f"Questions task failed: {e}")
                    processing_method['questions'] = "Failed"
        else:
            # Only summarization
            summary = summarize_with_api(text, max_length)
            processing_method['summary'] = "HuggingFace API" if summary else "Failed"
        
        # Fallback for summary only if API completely fails
        if not summary:
            logger.warning("Using fallback summary method")
            summary = simple_fallback_summary(text)
            processing_method['summary'] = "Simple fallback"
        
        # Ensure we have valid questions array
        if include_questions and not questions:
            questions = []
            processing_method['questions'] = "No questions generated"
        
        # Prepare response in format expected by ai.ts
        response = {
            "summary": summary,
            "questions": questions if include_questions else [],
            "method": processing_method.get('summary', 'Unknown'),
            "question_method": processing_method.get('questions', 'Not requested') if include_questions else 'Not requested',
            "original_length": len(text),
            "summary_length": len(summary) if summary else 0,
            "compression_ratio": f"{len(summary)/len(text)*100:.1f}%" if summary else "0%",
            "question_count": len(questions) if include_questions else 0,
            "status": "success"
        }
        
        logger.info(f"Processing completed: Summary={bool(summary)}, Questions={len(questions) if questions else 0}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "api_token_available": bool(HF_API_TOKEN),
        "models_status": "ready" if HF_API_TOKEN else "limited"
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Enhanced Text Processing API")
    print(f"📡 Port: {port}")
    print("🤖 Models:")
    print("   📝 Summarization: fransiskaarthaa/text-summarize")
    print("   ❓ Questions: meilanikizana/question-generation-indonesia")
    print("🔗 Main Endpoint: POST /summarize")
    
    if not HF_API_TOKEN:
        print("⚠️  Warning: No HF_TOKEN set. Limited functionality available.")
        print("   Add HF_TOKEN environment variable for full API access")
    else:
        print("✅ HF_TOKEN configured - Full API functionality available")
    
    app.run(host='0.0.0.0', port=port, debug=False)
