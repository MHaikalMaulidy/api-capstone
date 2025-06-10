from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Load Hugging Face clients
summarizer = Client("fransiskaarthaa/SUMMARIZE-MODEL")
question_generator = Client("meilanikizana/indonesian-question-generator")

@app.route("/summarize", methods=["POST"])
def summarize_text():
    data = request.get_json()
    input_text = data.get("text")

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    try:
        result = summarizer.predict(
            text=input_text,
            max_length=150,
            min_length=30,
            api_name="/summarize_text"
        )
        return jsonify({"summary": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-question", methods=["POST"])
def generate_question():
    data = request.get_json()
    input_text = data.get("text")
    num_questions = data.get("num_questions", 1)

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    try:
        result = question_generator.predict(
            context=input_text,
            num_questions=num_questions,
            api_name="/predict"
        )
        return jsonify({"questions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process-text", methods=["POST"])
def process_text():
    data = request.get_json()
    input_text = data.get("text")
    num_questions = data.get("num_questions", 1)

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    try:
        summary = summarizer.predict(
            text=input_text,
            max_length=150,
            min_length=30,
            api_name="/summarize_text"
        )
        questions = question_generator.predict(
            context=input_text,
            num_questions=num_questions,
            api_name="/predict"
        )

        return jsonify({
            "summary": summary,
            "questions": questions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Hugging Face API Flask App is running!", 200

if __name__ == "__main__":
    app.run(debug=True)
