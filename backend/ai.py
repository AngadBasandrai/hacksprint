import json
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import re

load_dotenv()

API_KEYS = os.getenv("API_KEYS", "").split(",")
API_KEYS = [k.strip() for k in API_KEYS if k.strip()]
if not API_KEYS:
    raise ValueError("API_KEYS list is empty. Please set the API_KEYS environment variable.")

SYSTEM_PROMPT = """You are an educational assistant AI.
Your job is to respond as accurately as possible.
If information is debatable, clearly mention that fact.

Your answers must follow this structure:
1. Break down the topic into multiple parts:
    a. Foundational principles/prerequisites
    b. Basic conceptual principles
    c. Advanced details / examples / research
    d. Further reading
2. Generate a study plan or roadmap.
3. Highlight important key concepts, formulae, and definitions.
4. Provide practice problems to test understanding.

Other instructions:
- Be clear, logical, and beginner-friendly unless a higher level is requested.
- Avoid unnecessary fluff or filler.
- Prioritize comprehension and factual accuracy.
- Provide mermaid diagrams where applicable (using the 'mermaid_diagram' field).
- When generating practice problems, ensure they cover all key concepts.
- Tailor difficulty level and age group based on user request (look for Control Parameters).

Output format:
Respond ONLY in a valid JSON object conforming to the given schema.
Do NOT include any explanations, markdown formatting, or commentary outside the JSON.
Always use double quotations.
Ensure all strings are properly escaped (use \\" for quotes, \\n for newlines, \\\\ for backslashes).
Do not write anything other than properly escaped code in the code section.
Always have escape characters in latex formulas.
Keep responses concise to avoid truncation.
Do not use brackets or special characters inside Mermaid code. Only use letters and simple connections.
For mermaid diagrams, use simple flowchart syntax: A --> B --> C (avoid complex syntax that might cause errors).
"""


current_key = 0

SCHEMA = {
    "type": "object",
    "properties": {
        "foundations": {"type": "string"},
        "concepts": {"type": "string"},
        "formulas": {"type": "string"},
        "keyconcepts": {"type": "string"},
        "problems": {"type": "string"},
        "study_plan": {"type": "string"},
        "further_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "mermaid_diagram": {"type": "string"},
        "code": {"type": "string"}
    },
    "required": [
        "foundations",
        "concepts",
        "formulas",
        "keyconcepts",
        "problems",
        "study_plan",
        "further_questions",
        "mermaid_diagram",
        "code"
    ],
}


def _get_client():
    return genai.Client(api_key=API_KEYS[current_key])


def _rotate_key():
    global current_key
    current_key = (current_key + 1) % len(API_KEYS)
    print(f"[API KEY ROTATION] -> Now using key #{current_key} / Total Keys: {len(API_KEYS)}")


def _extract_json_from_text(text):
    if not text:
        return None
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return text


def _fix_unescaped_characters(json_str):
    def fix_newlines_in_strings(match):
        content = match.group(1)
        content = content.replace('\n', '\\n').replace('\r', '\\r')
        return f'"{content}"'

    json_str = re.sub(r'"((?:[^"\\]|\\.)*)(?="|$)', lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r'), json_str)

    return json_str


def _repair_truncated_json(json_str, error_pos=None):
    print("[REPAIR] Attempting to fix truncated JSON...")

    fixed = json_str.rstrip()

    if not fixed.endswith('"') and not fixed.endswith('}') and not fixed.endswith(']'):

        quote_count = fixed.count('"') - fixed.count('\\"')
        if quote_count % 2 == 1:  # Odd number means unterminated string
            fixed += '"'
            print("[REPAIR] Added closing quote")

    open_brackets = fixed.count('[') - fixed.count(']')
    for _ in range(open_brackets):
        fixed += ']'
        print(f"[REPAIR] Added closing bracket (remaining: {open_brackets - _})")

    open_braces = fixed.count('{') - fixed.count('}')
    for _ in range(open_braces):
        fixed += '}'
        print(f"[REPAIR] Added closing brace (remaining: {open_braces - _})")

    return fixed


def _validate_and_fix_json(raw_text):
    try:
        parsed = json.loads(raw_text)
        print("[VALIDATION] JSON is valid as-is")
        return parsed
    except json.JSONDecodeError as e:
        print(f"[JSON ERROR] {e}")

    # Step 2: Try extracting from markdown
    extracted = _extract_json_from_text(raw_text)
    if extracted and extracted != raw_text:
        try:
            parsed = json.loads(extracted)
            print("[VALIDATION] Extracted JSON from markdown")
            return parsed
        except json.JSONDecodeError:
            raw_text = extracted  # Continue with extracted version

    # Step 3: Fix common escape issues
    try:
        fixed = _fix_unescaped_characters(raw_text)
        parsed = json.loads(fixed)
        print("[VALIDATION] Fixed unescaped characters")
        return parsed
    except json.JSONDecodeError as e:
        pass

    # Step 4: Try to repair truncated JSON
    try:
        repaired = _repair_truncated_json(raw_text, e.pos if 'e' in locals() else None)
        parsed = json.loads(repaired)
        print("[VALIDATION] Repaired truncated JSON")
        return parsed
    except json.JSONDecodeError:
        pass

    # Step 5: Last resort - try to salvage partial data
    print("[VALIDATION] Attempting partial data recovery...")
    try:
        # Try to extract at least some valid JSON objects
        partial_match = re.search(r'\{[^{}]*"foundations"[^{}]*\}', raw_text, re.DOTALL)
        if partial_match:
            # Build a minimal valid response
            fallback = {
                "foundations": "Error: Incomplete response from API",
                "concepts": "Error: Incomplete response from API", 
                "formulas": "N/A",
                "keyconcepts": "Error: Incomplete response from API",
                "problems": "N/A",
                "study_plan": "N/A",
                "further_questions": [],
                "mermaid_diagram": "graph TD\n    A[Response Error] --> B[Please regenerate]",
                "code": ""
            }
            print(f"[VALIDATION] Returning fallback response: {json.dumps(fallback, indent=2)}")
            return fallback
    except Exception:
        pass

    # If all else fails, raise the original error
    raise ValueError(f"Could not parse or repair JSON response. Length: {len(raw_text)}")


def _ensure_schema_compliance(data):
    """Ensure the parsed data contains all required fields from the schema."""
    required_fields = SCHEMA["required"]

    for field in required_fields:
        if field not in data:
            print(f"[SCHEMA FIX] Missing required field: {field}")
            if field == "further_questions":
                data[field] = []
            else:
                data[field] = ""

    # Ensure further_questions is a list
    if "further_questions" in data and not isinstance(data["further_questions"], list):
        print("[SCHEMA FIX] Converting further_questions to list")
        data["further_questions"] = [str(data["further_questions"])]

    return data


def _normalize_json_response(data):
    """Normalize and clean the JSON response."""

    # Ensure all string fields are actually strings
    for key, value in data.items():
        if key != "further_questions" and value is not None and not isinstance(value, str):
            data[key] = str(value)

    # Clean up any remaining control characters
    for key in data:
        if isinstance(data[key], str):
            # Remove control characters except newlines and tabs
            data[key] = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', data[key])

    return data


def ai(prompt, schema=SCHEMA, use_search=False, age=None, difficulty_level=None, max_retries=3):
    config_params = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.3,
        "max_output_tokens": 8192,
    }

    if use_search:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        config_params["tools"] = [grounding_tool]

    else:
        config_params["response_mime_type"] = "application/json"
        config_params["response_schema"] = schema

    config = types.GenerateContentConfig(**config_params)

    control_tokens = []
    if age is not None:
        control_tokens.append(f"Age Group: {age}")
    if difficulty_level is not None:
        control_tokens.append(f"Difficulty: {difficulty_level}")

    if control_tokens:
        control_string = "\n\nControl Parameters: " + ", ".join(control_tokens)
        final_prompt = prompt + control_string
    else:
        final_prompt = prompt

    total_attempts = len(API_KEYS) * max_retries
    attempt_count = 0

    while attempt_count < total_attempts:
        client = _get_client()
        try:
            print(f"Attempting API call #{attempt_count + 1}/{total_attempts} (Key #{current_key}). Search={use_search}")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=config,
                contents=final_prompt
            )

            # Check if response is None or empty
            if response is None:
                print("[ERROR] Response is None")
                attempt_count += 1
                _rotate_key()
                continue

            raw_text = response.text

            if raw_text is None:
                print("[ERROR] response.text is None")
                attempt_count += 1
                _rotate_key()
                continue

            raw_text = raw_text.strip()

            if use_search:
                print("Search Mode: Success.")
                return response

            else:
                print(f"JSON Mode: Received {len(raw_text)} characters")

                # Parse and validate JSON
                parsed = _validate_and_fix_json(raw_text)

                # Ensure schema compliance
                parsed = _ensure_schema_compliance(parsed)

                # Normalize the response
                parsed = _normalize_json_response(parsed)

                # Final validation - ensure it's serializable
                try:
                    json.dumps(parsed)
                    print(f"JSON Mode: Success. Response validated and normalized.")
                    return parsed
                except Exception as e:
                    print(f"[VALIDATION ERROR] Final serialization check failed: {e}")
                    attempt_count += 1
                    if attempt_count < total_attempts:
                        _rotate_key()
                    continue

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[JSON ERROR] {e}")
            if 'raw_text' in locals():
                print(f"[DEBUG] First 500 chars: {raw_text[:500]}")
                print(f"[DEBUG] Last 500 chars: {raw_text[-500:]}")
            attempt_count += 1
            if attempt_count < total_attempts:
                _rotate_key()

        except Exception as e:
            code = getattr(e, "code", None)

            if code in [401, 403, 429]:
                print(f"[ERROR {code}] API Key failed or Rate Limit exceeded.")
                _rotate_key()
                attempt_count += 1

            elif "responseSchema" in str(e) or ("tools" in str(e) and use_search is False):
                print("!!! [CRITICAL ERROR] Configuration Conflict: Cannot use tools (Search) with structured output.")
                raise e

            elif attempt_count >= total_attempts - 1:
                print(f"!!! [FATAL] All attempts failed. Last error: {e}")
                raise e

            else:
                print(f"[WARN] Error: {e}. Retrying...")
                _rotate_key()
                attempt_count += 1

    # Final fallback - return error response in valid format
    error_response = {
        "error": "All API attempts failed.",
        "foundations": "Error: Could not generate response",
        "concepts": "Error: Could not generate response",
        "formulas": "N/A",
        "keyconcepts": "Error: Could not generate response",
        "problems": "N/A",
        "study_plan": "N/A",
        "further_questions": [],
        "mindmap": "",
        "code": ""
    }
    print(f"[FINAL FALLBACK] Returning error response:\n{json.dumps(error_response, indent=2)}")
    return error_response