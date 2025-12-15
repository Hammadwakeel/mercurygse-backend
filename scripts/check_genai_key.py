#!/usr/bin/env python3
"""Quick script to validate Google GenAI API key and model access.

Run from repository root (inside venv) like:

    python3 scripts/check_genai_key.py

It will attempt to initialize the genai client using the same code in
`app.services.model_client` and make a lightweight request to validate the
key. The script prints a clear message for success / failure and includes
any provider error details.
"""
import importlib
import traceback
import sys

print("Checking GenAI API key and model access...")

try:
    mc = importlib.import_module("app.services.model_client")
except Exception as e:
    print("Failed to import app.services.model_client:", e)
    traceback.print_exc()
    sys.exit(2)

# Try to initialize client (reads env var GOOGLE_API_KEY)
client = None
try:
    c = mc.init_genai_client()
    # the init_genai_client returns the client and also assigns mc.genai_client
    client = c or getattr(mc, "genai_client", None)
except Exception as e:
    print("init_genai_client raised exception:", e)
    traceback.print_exc()
    sys.exit(2)

if not client:
    print("No GenAI client configured. Please set GOOGLE_API_KEY in environment.")
    sys.exit(1)

print("GenAI client created. Attempting a lightweight API call to verify key and model access...")

# Try to call a safe method. We attempt to use models.list() or models.get()
# if available, otherwise fall back to a small generate_content call.
try:
    models_api = getattr(client, "models", None)
    if models_api is None:
        print("Client has no .models attribute; cannot proceed.")
        sys.exit(3)

    # Prefer listing models if available
    if hasattr(models_api, "list"):
        try:
            res = models_api.list()
            print("Models list call succeeded. Sample output:")
            print(res)
            sys.exit(0)
        except Exception as e:
            print("models.list() failed (continuing to try other checks):", e)

    if hasattr(models_api, "get"):
        try:
            # Try to fetch a commonly available model
            model_name = "gemini-2.0-flash"
            res = models_api.get(model=model_name)
            print(f"models.get('{model_name}') succeeded:")
            print(res)
            sys.exit(0)
        except Exception as e:
            print("models.get() failed (continuing to try generate_content):", e)

    # Fallback: small generate_content call (may consume quota)
    # Use a very small prompt
    try:
        prompt = "Ping"
        print("Calling models.generate_content with a tiny prompt (may hit quota)...")
        resp = models_api.generate_content(model="gemini-2.0-flash", contents=[{"type": "text", "text": prompt}])
        text = getattr(resp, "text", None) or str(resp)
        print("generate_content succeeded, response preview:")
        print(text[:1000])
        sys.exit(0)
    except Exception as e:
        print("generate_content failed:", e)
        traceback.print_exc()
        # Inspect exception for structured error info
        try:
            err_str = str(e)
            print("Exception text:\n", err_str)
        except Exception:
            pass
        sys.exit(3)

except Exception as e:
    print("Unexpected error while validating key:", e)
    traceback.print_exc()
    sys.exit(2)
