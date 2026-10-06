import json
import re
from ollama import Client
from .config import OLLAMA_URL, OLLAMA_MODEL
from .tools import vector_search_tool, calculator_tool, python_interpreter_tool

TOOL_LABELS = {
    "vector_search": "RAG Search",
    "calculator": "Calculator",
    "python": "Python Interpreter Tool",
    "direct": "Direct Answer",
}

ROUTER_SYSTEM = """You are the routing agent for a technical-document assistant.

Your job is to identify the BEST TOOL for the user's CURRENT REQUEST before answering.

Available tools:

1. python
Use the Python Interpreter Tool / Code Runner when the user asks to:
- write Python code
- generate Python code
- create a Python script
- implement something in Python
- solve a programming problem using Python
- show a Python example
- write a Python function
- write Python classes
- write Python automation code
- modify or debug Python code
- explain Python code with an example
- execute or calculate something using Python
- create code using asyncio, threading, multiprocessing, pandas, numpy, etc.
- produce a Python implementation based on a requirement

IMPORTANT:
If the user's main request is to WRITE or GENERATE PYTHON CODE,
choose python even if the topic is related to the Python documentation.

2. vector_search
Use RAG Search when the user asks for information that must be
retrieved from the supplied documents, for example:
- What does the Python documentation say about X?
- What does the AWS Well-Architected Framework say about X?
- What does NIST SP 800-145 define as X?
- Find a specific definition, rule, requirement, or fact in the documents.
- Questions requiring document citations.

If the user only wants Python code, do NOT choose vector_search.

3. calculator
Use Calculator for arithmetic or mathematical expressions, for example:
- 25 * 40
- 1250 * 0.18 + 47.5
- calculate a percentage
- perform a numerical calculation

4. direct
Use Direct Answer for:
- greetings
- casual conversation
- simple explanations that do not require document retrieval
- non-tool questions

IMPORTANT ROUTING RULES:

A. "Write Python code to..."
   -> python

B. "Give me a Python example for..."
   -> python

C. "Create a Python function that..."
   -> python

D. "Implement this in Python..."
   -> python

E. "Debug this Python code..."
   -> python

F. "How do I do X in Python? Show code."
   -> python

G. "According to the Python documentation, what is asyncio.TaskGroup?"
   -> vector_search

H. "What does the Python documentation say about asyncio.TaskGroup?"
   -> vector_search

I. "Look up asyncio.TaskGroup in the documentation and explain it."
   -> vector_search

J. "Look up asyncio.TaskGroup and then write Python code using it."
   -> python
   because the primary requested output is Python code.

Always prioritize the user's requested OUTPUT TYPE.
If the requested output is Python code, choose python.

Return ONLY valid JSON:

{
  "tool": "vector_search|calculator|python|direct",
  "input": "...",
  "reason": "short reason"
}

Never return markdown.
Never invent document facts.
"""

ANSWER_SYSTEM = """You are the final answer agent for a technical RAG assistant.
Use retrieved document context for document facts. Cite document claims inline as [Document, p. N].
Be clear and concise. If code is requested, give a minimal working example.
Do not claim a tool was used unless it actually ran.
"""

def _client():
    return Client(host=OLLAMA_URL)


def is_python_code_request(query: str) -> bool:
    """
    Detect whether the user's primary request is to write,
    generate, implement, modify, debug, or execute Python code.

    This rule intentionally takes priority over documentation
    lookup when the user asks for code as the final output.
    """

    q = query.lower().strip()

    # Strong Python/code-generation phrases
    code_phrases = [
        "write python code",
        "write python program",
        "write python script",
        "write a python function",
        "write python function",
        "create python code",
        "create a python program",
        "create a python script",
        "generate python code",
        "generate a python program",
        "generate a python script",
        "implement in python",
        "implement this in python",
        "code in python",
        "python example",
        "python code example",
        "python implementation",
        "python program",
        "python script",
        "python function",
        "python class",
        "python code",
        "debug python",
        "modify python code",
        "fix python code",
        "execute python",
        "run python code",
    ]

    # Documentation + code combination
    documentation_code_phrases = [
        "look up",
        "lookup",
        "according to the documentation",
        "python documentation",
        "python reference",
        "reference documentation",
    ]

    code_output_phrases = [
        "write",
        "create",
        "generate",
        "implement",
        "show",
        "give me",
        "provide",
        "build",
    ]

    # --------------------------------------------------
    # Rule 1:
    # Direct Python code request
    # --------------------------------------------------

    if any(phrase in q for phrase in code_phrases):
        return True

    # --------------------------------------------------
    # Rule 2:
    # Documentation lookup + code generation
    #
    # Example:
    # "Look up asyncio.TaskGroup in the Python
    # reference documentation and write a code example."
    # --------------------------------------------------

    has_documentation_request = any(
        phrase in q
        for phrase in documentation_code_phrases
    )

    has_code_output_request = any(
        phrase in q
        for phrase in code_output_phrases
    )

    mentions_python = "python" in q or "asyncio" in q

    if (
        has_documentation_request
        and has_code_output_request
        and mentions_python
    ):
        return True

    # --------------------------------------------------
    # Rule 3:
    # Python-specific programming concepts + code request
    # --------------------------------------------------

    python_concepts = [
        "asyncio",
        "taskgroup",
        "coroutine",
        "threading",
        "multiprocessing",
        "pandas",
        "numpy",
        "flask",
        "fastapi",
        "django",
        "pytest",
    ]

    has_python_concept = any(
        concept in q
        for concept in python_concepts
    )

    if has_python_concept and has_code_output_request:
        return True

    return False


def route(query: str, history=None):
    """
    Route the user's request to the correct tool.

    Python code-generation requests are handled deterministically
    before asking the LLM router. This prevents the LLM from
    incorrectly choosing RAG when the user asks for code based
    on documentation.
    """

    # =========================================================
    # PRIORITY 1 — PYTHON CODE REQUEST
    # =========================================================

    if is_python_code_request(query):

        return {
            "tool": "python",
            "input": query,
            "reason": (
                "The user requested Python code as the primary output. "
                "Python Interpreter Tool has priority over documentation lookup."
            ),
        }

    # =========================================================
    # PRIORITY 2 — NORMAL LLM ROUTING
    # =========================================================

    client = _client()

    messages = [
        {
            "role": "system",
            "content": ROUTER_SYSTEM
        }
    ]

    if history:

        for m in history[-6:]:

            if m.get("role") in ("user", "assistant"):

                messages.append(
                    {
                        "role": m["role"],
                        "content": str(
                            m.get("content", "")
                        )
                    }
                )

    messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": 0
        }
    )

    content = response["message"]["content"].strip()

    match = re.search(
        r"\{.*\}",
        content,
        re.S
    )

    # Fallback
    if not match:

        return {
            "tool": "vector_search",
            "input": query,
            "reason": "Document-first fallback"
        }

    try:

        data = json.loads(
            match.group(0)
        )

        if data.get("tool") not in TOOL_LABELS:

            raise ValueError(
                "Invalid tool selected"
            )

        return data

    except Exception:

        return {
            "tool": "vector_search",
            "input": query,
            "reason": "Document-first fallback"
        }

def answer(query: str, history=None, selected_tool=None):
    plan = route(query, history) if not selected_tool else {
        "tool": selected_tool,
        "input": query,
        "reason": "Tool selected by the routing agent before execution"
    }
    tool = plan["tool"]
    tool_input = plan.get("input") or query
    steps = [f"Agent selected: {TOOL_LABELS[tool]} — {plan.get('reason', '')}"]
    citations = []

    if tool == "vector_search":
        result = vector_search_tool(tool_input)
        rows = result.get("results", [])
        steps.append(f"RAG Search retrieved {len(rows)} document chunks.")
        citations = [
            {
                "document": r["source_doc"],
                "page": r["page_number"],
                "section": r.get("section_heading")
            }
            for r in rows
        ]
        context = "\n\n".join(
            f"[{r['source_doc']}, p. {r['page_number']}]\n{r['text']}"
            for r in rows
        )
        prompt = f"Question: {query}\n\nRetrieved context:\n{context}\n\nAnswer with citations."

    elif tool == "calculator":
        try:
            result = calculator_tool(tool_input)
            steps.append("Calculator executed the numeric expression.")
        except Exception as e:
            result = f"Calculator error: {e}"
        prompt = f"Question: {query}\nCalculator result: {result}"

    elif tool == "python":
        try:
            result = python_interpreter_tool(tool_input)
            steps.append("Python tool executed the requested safe code.")
        except Exception as e:
            result = f"Python tool error: {e}"
        prompt = f"Question: {query}\nPython tool result: {result}"

    else:
        result = "No external tool required."
        steps.append("No external tool was required.")
        prompt = f"Question: {query}"

    client = _client()
    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    if history:
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": str(m.get("content", ""))})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.2}
    )
    final_answer = response["message"]["content"].strip()

    seen, unique = set(), []
    for c in citations:
        key = (c["document"], c["page"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return {
        "answer": final_answer,
        "tool": tool,
        "tool_label": TOOL_LABELS[tool],
        "citations": unique,
        "steps": steps,
    }
