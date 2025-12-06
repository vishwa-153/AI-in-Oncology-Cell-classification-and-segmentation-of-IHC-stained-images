import base64
import requests
import json
import os

# === CONFIGURATION ===
# Do NOT hardcode keys. Pull from environment variables set in HF Space settings.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Set it in your Space Secrets as GEMINI_API_KEY.")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"


# === FUNCTION TO ENCODE IMAGE TO BASE64 ===
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# === FUNCTION TO SEND REQUEST TO GEMINI ===
def query_gemini_with_image_and_prompt(image_path, prompt):
    b64_image = encode_image_to_base64(image_path)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image
                        }
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "[⚠️] Gemini returned an unexpected response format."
    else:
        return f"[❌] Error {response.status_code}: {response.text}"
def LLM_Analysis(image,predicted,confidence):
    image_file=image
    answer=predicted
    prompt_text = f"You are a board-certified surgical pathologist reviewing an IHC-stained tissue section with overlaid segmentation. The AI model has classified this field as “{predicted}” with a confidence of {confidence}. Please provide a one-paragraph diagnostic interpretation in precise medical terminology, including at least three quantitative metrics (e.g., percentage of positive cells, mean nuclear diameter, DAB intensity values) that support or contextualize the prediction. Avoid vague or repetitive language."
    result = query_gemini_with_image_and_prompt(image_file, prompt_text)
    return result

# === MAIN EXECUTION ===
if __name__ == "__main__":

    image_file = "output.png"  

    answer="High grade"
    prompt_text = "You are a board-certified surgical pathologist reviewing an IHC-stained lung/colon tissue section with overlaid segmentation. The AI model has classified this field as “{PREDICTED_OUTPUT}” with a confidence of {MODEL_CONFIDENCE}. Please provide a one-paragraph diagnostic interpretation in precise medical terminology, including at least three quantitative metrics (e.g., percentage of positive cells, mean nuclear diameter, DAB intensity values) that support or contextualize the prediction. Avoid vague or repetitive language."

    print("Analysing...")
    result = query_gemini_with_image_and_prompt(image_file, prompt_text)
    print("\nReport:\n")
    print(result)
