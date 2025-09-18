import os
import requests
import json

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

EXPANSION_PROMPT = (
    "Expand the following user interest into a nonprofit-style mission statement paragraph:\n"
    "User interest: \"{interest}\"\n"
    "Return only the final mission statement without any additional commentary or reasoning"
)

def expand_interest(interest: str) -> str:
    prompt = EXPANSION_PROMPT.format(interest=interest)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_KEY}"
    }
    payload = json.dumps({
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        # "max_tokens": 50,
        # "temperature": 0.7
    })
    response = requests.post(OPENROUTER_URL, data=payload, headers=headers)
    # print("Raw response:", response.text)

    # print("Final output:", response.json())
    try:
        response.raise_for_status()
        resp_json = response.json()
        return resp_json["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as e:
        print("Falling back to raw interests. Error response:", response.text)
        return(','.join(interest))  # Fallback to original interest on error
    