import requests
import json
import socket
import os



# CHANGE THIS if needed
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://140.105.52.180:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

SYSTEM_PROMPT = "You are a helpful assistant."

def call_ollama(user_message: str, model: str, temperature: float = 0.7):
    """
    Minimal test call to Ollama on Spark.
    """

    payload = {
        "model": model,
        "prompt": f"""{SYSTEM_PROMPT}

User: {user_message}
Assistant:""",
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    print(f"Sending request to {OLLAMA_GENERATE_URL}...")

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("\n❌ HTTP error from Ollama:")
        try:
            err = response.json().get("error", str(e))
            print(err)
            if "not found" in err and "model" in err:
                print(f"\nTry pulling a model first, for example:\n  ollama pull {model}")
        except Exception:
            print(e)
        return
    except Exception as e:
        print("\n❌ Connection error:")
        print(e)
        return

    data = response.json()

    print("\n✅ Raw response:")
    print(data)

    if "response" not in data:
        print("\n❌ Unexpected format. Full payload:")
        print(data)
        return

    print("\n🧠 Model output:\n")
    print(data["response"])


if __name__ == "__main__":
    print("=== OLLAMA TEST START ===")
    print("HOSTNAME:", socket.gethostname())
    print("NODE:", os.uname())
    print("LOCAL IP:", socket.gethostbyname(socket.gethostname()))
    print("OLLAMA BASE URL:", OLLAMA_BASE_URL)

    # 1. Check server visibility
    try:
        tags = requests.get(OLLAMA_TAGS_URL, timeout=10).json()
        print("\n📦 Available models:")
        print(json.dumps(tags, indent=2))
    except Exception as e:
        print("\n❌ Cannot reach Ollama API:")
        print(e)
        exit(1)

    available_models = [m.get("name") for m in tags.get("models", []) if m.get("name")]
    if not available_models:
        print("\n❌ No local Ollama models are installed.")
        print("Pull one first, for example:")
        print("  ollama pull llama3.2")
        exit(1)

    requested_model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    if requested_model not in available_models:
        fallback_model = available_models[0]
        print(f"\n⚠️ Requested model '{requested_model}' is not installed.")
        print(f"Using installed model '{fallback_model}' instead.")
        requested_model = fallback_model

    # 2. Test generation
    call_ollama("Say hello in one sentence.", model=requested_model)