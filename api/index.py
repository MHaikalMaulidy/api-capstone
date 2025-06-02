from flask import Flask, request, jsonify
import requests
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Flask
app = Flask(__name__)

# Hugging Face API Configuration
HF_API_TOKEN = os.environ.get('HF_TOKEN')  # Set di Replit Secrets
HF_API_URL = "https://api-inference.huggingface.co/models/fransiskaarthaa/text-summarize"

def summarize_with_api(text, max_length=150):
    """Gunakan Hugging Face Inference API - tidak perlu download model"""
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
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('summary_text', '')
            return result.get('summary_text', '')
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Request error: {e}")
        return None

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
        "message": "Lightweight Text Summarization API",
        "status": "running",
        "methods": ["HuggingFace API", "Simple fallback"],
        "storage_usage": "Minimal - no local models"
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Endpoint summarization"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Field 'text' required"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) < 50:
            return jsonify({"error": "Text too short for summarization"}), 400
            
        max_length = min(data.get('max_length', 150), 500)  # Limit max
        use_api = data.get('use_api', True)
        
        summary = None
        method_used = ""
        
        # Try HuggingFace API first
        if use_api:
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

@app.route('/health', methods=['GET'])
def health():
    """Health check untuk monitoring"""
    return jsonify({"status": "healthy", "storage": "minimal"})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Lightweight Summarization API")
    print(f"📡 Port: {port}")
    print("💾 Storage usage: MINIMAL (no local models)")
    print("🔗 Endpoints:")
    print("   GET  / - Status")
    print("   POST /summarize - Summarize text") 
    print("   GET  /health - Health check")
    
    if not HF_API_TOKEN:
        print("⚠️  Warning: No HF_TOKEN set. API method may be limited.")
        print("   Add HF_TOKEN in Replit Secrets for full functionality")
    
    app.run(host='0.0.0.0', port=port, debug=False)
