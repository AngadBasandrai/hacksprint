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

SIMPLE_DIAGRAM_PROMPT = """Generate a VERY SIMPLE mermaid flowchart diagram about: {topic}

CRITICAL RULES:
1. Use ONLY basic flowchart syntax: graph TD
2. Use ONLY simple arrows: -->
3. Use ONLY letters for node IDs (A, B, C, etc.)
4. Keep node labels SHORT (max 20 characters)
5. Use ONLY alphanumeric characters and spaces in labels
6. Maximum 5-6 nodes total
7. NO special characters, NO brackets in labels
8. NO quotes around labels unless absolutely necessary

Example format:
graph TD
    A[Start] --> B[Step One]
    B --> C[Step Two]
    C --> D[End]

Return ONLY the mermaid code, nothing else."""


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
        if quote_count % 2 == 1:
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

    extracted = _extract_json_from_text(raw_text)
    if extracted and extracted != raw_text:
        try:
            parsed = json.loads(extracted)
            print("[VALIDATION] Extracted JSON from markdown")
            return parsed
        except json.JSONDecodeError:
            raw_text = extracted

    try:
        fixed = _fix_unescaped_characters(raw_text)
        parsed = json.loads(fixed)
        print("[VALIDATION] Fixed unescaped characters")
        return parsed
    except json.JSONDecodeError as e:
        pass

    try:
        repaired = _repair_truncated_json(raw_text, e.pos if 'e' in locals() else None)
        parsed = json.loads(repaired)
        print("[VALIDATION] Repaired truncated JSON")
        return parsed
    except json.JSONDecodeError:
        pass

    print("[VALIDATION] Attempting partial data recovery...")
    try:
        partial_match = re.search(r'\{[^{}]*"foundations"[^{}]*\}', raw_text, re.DOTALL)
        if partial_match:
            fallback = {
                "foundations": "Error: Incomplete response from API",
                "concepts": "Error: Incomplete response from API", 
                "formulas": "N/A",
                "keyconcepts": "Error: Incomplete response from API",
                "problems": "N/A",
                "study_plan": "N/A",
                "further_questions": [],
                "mermaid_diagram": "",
                "code": ""
            }
            print(f"[VALIDATION] Returning fallback response: {json.dumps(fallback, indent=2)}")
            return fallback
    except Exception:
        pass

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

    if "further_questions" in data and not isinstance(data["further_questions"], list):
        print("[SCHEMA FIX] Converting further_questions to list")
        data["further_questions"] = [str(data["further_questions"])]

    return data


def _normalize_json_response(data):
    """Normalize and clean the JSON response."""
    for key, value in data.items():
        if key != "further_questions" and value is not None and not isinstance(value, str):
            data[key] = str(value)

    for key in data:
        if isinstance(data[key], str):
            data[key] = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', data[key])

    return data


def _generate_simple_diagram(topic, max_attempts=2):
    """Generate a simple mermaid diagram with retry logic"""
    print(f"[DIAGRAM] Attempting to generate simple diagram for topic: {topic}")
    
    for attempt in range(max_attempts):
        try:
            client = _get_client()
            prompt = SIMPLE_DIAGRAM_PROMPT.format(topic=topic)
            
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500,
            )
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                config=config,
                contents=prompt
            )
            
            if response and response.text:
                diagram = response.text.strip()
                # Clean up the diagram
                diagram = diagram.replace('```mermaid', '').replace('```', '').strip()
                print(f"[DIAGRAM] Successfully generated simple diagram (attempt {attempt + 1})")
                return diagram
                
        except Exception as e:
            print(f"[DIAGRAM] Attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                _rotate_key()
            
    print("[DIAGRAM] All attempts to generate simple diagram failed")
    return ""


def _validate_mermaid_diagram(diagram):
    """Check if a mermaid diagram is likely to be valid"""
    if not diagram or not diagram.strip():
        return False
    
    # Check for common error indicators
    error_indicators = [
        'Diagram Not Available',
        'Diagram Error',
        'Please try regenerating',
        'No diagram available',
        'Error',
        'Failed'
    ]
    
    for indicator in error_indicators:
        if indicator in diagram:
            return False
    
    # Check if it has basic mermaid structure
    diagram_lower = diagram.lower()
    valid_types = ['graph', 'flowchart', 'sequencediagram', 'classdiagram', 'pie', 'journey', 'gantt']
    
    has_valid_type = any(diagram_lower.startswith(t) for t in valid_types)
    has_arrows = '-->' in diagram or '---' in diagram
    
    return has_valid_type or has_arrows


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

                parsed = _validate_and_fix_json(raw_text)
                parsed = _ensure_schema_compliance(parsed)
                parsed = _normalize_json_response(parsed)

                # Validate and retry diagram generation if needed
                if "mermaid_diagram" in parsed:
                    if not _validate_mermaid_diagram(parsed["mermaid_diagram"]):
                        print("[DIAGRAM] Invalid diagram detected, attempting simple diagram generation")
                        # Extract topic from prompt for simple diagram
                        topic = prompt[:100] if len(prompt) > 100 else prompt
                        simple_diagram = _generate_simple_diagram(topic)
                        if simple_diagram and _validate_mermaid_diagram(simple_diagram):
                            parsed["mermaid_diagram"] = simple_diagram
                            print("[DIAGRAM] Successfully replaced with simple diagram")
                        else:
                            # Set to empty string to hide in frontend
                            parsed["mermaid_diagram"] = ""
                            print("[DIAGRAM] Could not generate valid diagram, setting to empty")

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

    error_response = {
        "error": "All API attempts failed.",
        "foundations": "Error: Could not generate response",
        "concepts": "Error: Could not generate response",
        "formulas": "N/A",
        "keyconcepts": "Error: Could not generate response",
        "problems": "N/A",
        "study_plan": "N/A",
        "further_questions": [],
        "mermaid_diagram": "",
        "code": ""
    }
    print(f"[FINAL FALLBACK] Returning error response:\n{json.dumps(error_response, indent=2)}")
    return error_response