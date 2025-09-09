import json
import os

# Import your existing heavy logic (adjust the import if your file is elsewhere)
from src import lambda_function

def _default_event():
    # Provide whatever your heavy function expects
    # You can decode JSON from env if you want to pass parameters into the task.
    payload = os.getenv("FC_EVENT_JSON")
    if payload:
        try:
            return json.loads(payload)
        except Exception:
            pass
    return {}

def main():
    # Call your heavy logic once; block until complete
    event = _default_event()
    lambda_function.lambda_handler(event, None)

if __name__ == "__main__":
    main()
