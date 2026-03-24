from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
import re
import http.cookiejar

app = Flask(__name__)
CORS(app, origins="*")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  #add comma and then paste the api key from console.groq in "double quotes" in the brackets() #
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
COOKIE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")

def make_ytt():
    from youtube_transcript_api import YouTubeTranscriptApi
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    if os.path.exists(COOKIE_PATH):
        jar = http.cookiejar.MozillaCookieJar(COOKIE_PATH)
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies = jar
    try:
        return YouTubeTranscriptApi(http_client=session)
    except TypeError:
        return YouTubeTranscriptApi()

def fetch_best_transcript(video_id):
    """
    Try multiple strategies to get a transcript.
    Works for regular videos, Shorts, and recorded live streams.
    Returns (text, language) or raises Exception.
    """
    ytt = make_ytt()

    # Strategy 1: list all available transcripts and pick the best one
    # Priority: manual english > auto english > any manual > any auto > first available
    try:
        tlist = list(ytt.list(video_id))
        print(f"[TRANSCRIPT] Found {len(tlist)} transcript(s)")

        manual_en, auto_en, manual_any, auto_any = None, None, None, None

        for t in tlist:
            lang = t.language_code.lower()
            is_generated = t.is_generated
            print(f"  → lang={lang}, generated={is_generated}")

            if lang.startswith("en") and not is_generated:
                manual_en = t
            elif lang.startswith("en") and is_generated:
                auto_en = t
            elif not is_generated and manual_any is None:
                manual_any = t
            elif is_generated and auto_any is None:
                auto_any = t

        # Pick best available
        chosen = manual_en or auto_en or manual_any or auto_any or (tlist[0] if tlist else None)

        if chosen:
            print(f"[TRANSCRIPT] Using: {chosen.language_code} (generated={chosen.is_generated})")
            fetched = chosen.fetch()
            text = " ".join([s.text for s in fetched])
            return text, chosen.language_code

    except Exception as e:
        print(f"[TRANSCRIPT] Strategy 1 (list) failed: {e}")

    # Strategy 2: direct fetch with language fallbacks
    for lang in [["en"], ["en-US"], ["en-GB"], None]:
        try:
            if lang:
                fetched = ytt.fetch(video_id, languages=lang)
            else:
                fetched = ytt.fetch(video_id)
            text = " ".join([s.text for s in fetched])
            print(f"[TRANSCRIPT] Strategy 2 OK (lang={lang})")
            return text, str(lang)
        except Exception as e:
            print(f"[TRANSCRIPT] Strategy 2 lang={lang} failed: {e}")

    # Strategy 3: try fetching via YouTube's timedtext API directly
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        page = requests.get(url, headers=headers, timeout=10)
        # Find caption tracks in page source
        matches = re.findall(r'"baseUrl":"(https://www\.youtube\.com/api/timedtext[^"]+)"', page.text)
        if matches:
            caption_url = matches[0].replace("\\u0026", "&")
            caption_res = requests.get(caption_url, headers=headers, timeout=10)
            # Parse XML captions
            texts = re.findall(r'<text[^>]*>([^<]+)</text>', caption_res.text)
            if texts:
                import html
                text = " ".join([html.unescape(t) for t in texts])
                print(f"[TRANSCRIPT] Strategy 3 (timedtext) OK — {len(text)} chars")
                return text, "en"
    except Exception as e:
        print(f"[TRANSCRIPT] Strategy 3 (timedtext) failed: {e}")

    raise Exception(
        "No transcript available for this video. This can happen with: "
        "(1) Shorts with no captions, "
        "(2) Live streams that haven't been processed yet, "
        "(3) Videos with captions disabled by the creator. "
        "Try a different video or enable captions on YouTube first."
    )

@app.route("/transcript", methods=["GET"])
def get_transcript():
    video_id = request.args.get("id")
    print(f"\n[TRANSCRIPT] ── Fetching: {video_id} ──")
    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400
    try:
        text, lang = fetch_best_transcript(video_id)
        print(f"[TRANSCRIPT] ✅ Success — {len(text)} chars, lang={lang}")
        return jsonify({"transcript": text, "language": lang})
    except Exception as e:
        print(f"[TRANSCRIPT] ❌ All strategies failed: {e}")
        return jsonify({"error": str(e)}), 500

# Models with their max safe prompt character limits
MODELS = [
    {"name": "llama-3.3-70b-versatile", "max_chars": 6000},
    {"name": "qwen/qwen3-32b",          "max_chars": 4000},
    {"name": "llama-3.1-8b-instant",    "max_chars": 2500},
]

def trim_prompt(prompt, max_chars):
    """Trim transcript portion of prompt to fit model limit."""
    if len(prompt) <= max_chars:
        return prompt
    # Find where transcript starts and trim only that part
    marker = "Transcript:"
    idx = prompt.rfind(marker)
    if idx != -1:
        header = prompt[:idx + len(marker) + 1]
        transcript = prompt[idx + len(marker) + 1:]
        allowed = max_chars - len(header) - 50
        trimmed = transcript[:max(allowed, 500)]
        return header + trimmed
    return prompt[:max_chars]

def call_groq(prompt, retries=4):
    """Call Groq API with automatic retry, model rotation and prompt trimming."""
    import time
    import re as _re

    for attempt in range(retries):
        m = MODELS[min(attempt, len(MODELS) - 1)]
        model = m["name"]
        safe_prompt = trim_prompt(prompt, m["max_chars"])
        print(f"[AI] Attempt {attempt+1} — model={model} prompt_len={len(safe_prompt)}")

        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": safe_prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.7
                },
                timeout=60
            )
            data = response.json()

            if response.status_code == 200:
                print(f"[AI] Success with {model}")
                return data["choices"][0]["message"]["content"], None

            err_msg = data.get("error", {}).get("message", "")
            print(f"[AI] Error ({response.status_code}): {err_msg}")

            # Rate limit or too large — wait then try next model
            if response.status_code in (429, 413):
                wait_match = _re.search(r"try again in (\d+\.?\d*)s", err_msg)
                wait = float(wait_match.group(1)) if wait_match else 20
                wait = min(wait, 40)
                print(f"[AI] Waiting {wait}s before next attempt...")
                time.sleep(wait)
                continue

            return None, err_msg

        except Exception as ex:
            print(f"[AI] Exception: {ex}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return None, str(ex)

    return None, "All models are currently busy. Please wait a moment and try again."

@app.route("/claude", methods=["POST"])
def claude_proxy():
    body = request.get_json()
    prompt = body.get("prompt", "")
    print(f"[AI] Prompt: {len(prompt)} chars")
    text, err = call_groq(prompt)
    if text:
        return jsonify({"text": text})
    return jsonify({"error": err}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "VidMind running!",
        "cookies": "FOUND" if os.path.exists(COOKIE_PATH) else "NOT FOUND"
    })

if __name__ == "__main__":
    print("=========================================")
    print("  VidMind Python server — port 5000")
    print("  Cookies:", "FOUND ✅" if os.path.exists(COOKIE_PATH) else "NOT FOUND ❌")
    print("=========================================")
    app.run(host="0.0.0.0", port=5000, debug=False)
