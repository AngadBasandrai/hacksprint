from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai import ai  # your AI wrapper
import json
import os
import re

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str

def sanitize_mermaid_text(text: str) -> str:
    # Replace HTML line breaks with newline
    text = re.sub(r'<br\s*/?>', r'\n', text, flags=re.IGNORECASE)
    # Remove other HTML tags
    text = re.sub(r'<.*?>', '', text)
    return text

def escape_mermaid_chars(text: str) -> str:
    replacements = {
        '{': '(',
        '}': ')',
        '&': 'and',
        '#': '',
        '%': 'percent',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def wrap_text(text: str, width: int = 40) -> str:
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) > width:
            lines.append(current_line.strip())
            current_line = word
        else:
            current_line += " " + word
    lines.append(current_line.strip())
    return "\n".join(lines)

def preprocess_mermaid(diagram: str) -> str:
    if not diagram or not diagram.strip():
        return "graph TD\n    A[No diagram available]"
    
    try:
        # Remove HTML tags
        diagram = sanitize_mermaid_text(diagram)
        diagram = escape_mermaid_chars(diagram)

        # Wrap node labels in quotes and safe line breaks
        def quote_node(match):
            content = match.group(1)
            content = ' '.join(content.split())
            content = wrap_text(content, width=35)  # wrap for better display
            return f'["{content}"]'

        diagram = re.sub(r'\[(.*?)\]', quote_node, diagram)
        
        # Basic validation - ensure it starts with a valid mermaid type
        diagram = diagram.strip()
        if not any(diagram.startswith(x) for x in ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'gitgraph', 'pie', 'journey', 'gantt']):
            # Add a default graph type if missing
            diagram = f"graph TD\n    {diagram}"
            
        return diagram
    except Exception as e:
        print(f"Error preprocessing mermaid diagram: {e}")
        # Return a safe fallback diagram
        return "graph TD\n    A[Diagram Error] --> B[Please try regenerating]"

def sanitize_ai_json(json_str: str) -> dict:
    json_str = json_str.replace('\n', '\\n').replace('\r', '')
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {json.dumps({'error': str(e)}, indent=2)}")
        json_str = re.sub(r'(?<!\\)"', '\\"', json_str)
        try:
            return json.loads(json_str)
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Failed to parse AI JSON: {ex}")

# === Routes ===
@app.get("/health")
def health():
    return {"status": "yo im fine"}

@app.get("/")
def root():
    return {"about": "created by datavorous"}

@app.post("/generate")
def generate(query: Prompt):
    try:
        raw_result = ai(query.prompt)
        result_json = sanitize_ai_json(raw_result) if isinstance(raw_result, str) else raw_result
    
        # Ensure mermaid_diagram exists and preprocess it
        if "mermaid_diagram" in result_json and result_json["mermaid_diagram"]:
            result_json["mermaid_diagram"] = preprocess_mermaid(result_json["mermaid_diagram"])
        else:
            # Provide fallback diagram if missing
            result_json["mermaid_diagram"] = "graph TD\n    A[Diagram Not Available]"

        print(f"[GENERATE] Response:\n{json.dumps(result_json, indent=2)}")
        return result_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/demo")
def demo(query: Prompt):
    if "write" in query.prompt:
        demo_file_path = "../demos/gc2.json"
    elif "garbage" in query.prompt:
        demo_file_path = "../demos/gc.json"
    elif "equa" in query.prompt:
        demo_file_path = "../demos/eqn_motion.json"
    elif "mughal" in query.prompt:
        demo_file_path = "../demos/mughal.json"
    elif "regression" in query.prompt:
        demo_file_path = "demos/regression.json"
    elif "maximum" in query.prompt:
        demo_file_path = "demos/regression2.json"
    else:
        demo_file_path = "../demos/error.json"

    if not os.path.exists(demo_file_path):
        raise HTTPException(status_code=404, detail="demo.json not found")
    try:
        with open(demo_file_path, "r", encoding="utf-8") as f:
            demo_data = json.load(f)

        if 'mermaid_diagram' in demo_data:
            demo_data['mermaid_diagram'] = preprocess_mermaid(demo_data['mermaid_diagram'])

        print(f"[DEMO] Response:\n{json.dumps(demo_data, indent=2)}")
        return demo_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in demo.json")