from flask import Flask, jsonify, request
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import os

app = Flask(__name__)
CORS(app, origins="*")

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.route("/transcript", methods=["GET"])
def get_transcript():
    video_id = request.args.get("id")
    print(f"[TRANSCRIPT] Fetching: {video_id}")
    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id)
        full_text = " ".join([snippet.text for snippet in transcript])
        print(f"[TRANSCRIPT] Success: {len(full_text)} chars")
        return jsonify({"transcript": full_text})
    except Exception as ex:
        try:
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(video_id)
            first = next(iter(transcript_list))
            fetched = first.fetch()
            full_text = " ".join([snippet.text for snippet in fetched])
            return jsonify({"transcript": full_text})
        except Exception as ex2:
            print(f"[TRANSCRIPT ERROR] {ex2}")
            return jsonify({"error": str(ex2)}), 500

@app.route("/claude", methods=["POST"])
def claude_proxy():
    body = request.get_json()
    prompt = body.get("prompt", "")
    print(f"[AI] Prompt length: {len(prompt)} chars")

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            },
            timeout=60
        )
        data = response.json()
        print(f"[AI] Response status: {response.status_code}")

        if response.status_code == 200:
            text = data["choices"][0]["message"]["content"]
            return jsonify({"text": text})
        else:
            print(f"[AI ERROR] {data}")
            return jsonify({"error": data.get("error", {}).get("message", "AI error")}), 500

    except Exception as ex:
        print(f"[AI EXCEPTION] {ex}")
        return jsonify({"error": str(ex)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "VidMind server running!", "ai": "Groq (llama3-8b-8192)"})

if __name__ == "__main__":
    print("=========================================")
    print("  VidMind Python server on port 5000")
    print("  AI: Groq free tier (llama3-8b)")  
    print("=========================================")
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("  ⚠️  Set your GROQ_API_KEY in server.py!")
    else:
        print("  ✅ Groq API key loaded")
    print("=========================================")
    app.run(host="0.0.0.0", port=5000, debug=False)