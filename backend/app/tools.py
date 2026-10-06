import ast
import math
import operator as op
from sentence_transformers import SentenceTransformer
from .ingestion import get_collection
from .config import EMBEDDING_MODEL, TOP_K

_EMBEDDER = None

def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDER

def vector_search_tool(query: str, top_k: int = TOP_K) -> dict:
    collection = get_collection()
    if collection.count() == 0:
        return {"results": [], "message": "Vector index is empty. Run ingestion first."}

    vector = embedder().encode([query], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        rows.append({
            "text": doc,
            "source_doc": meta.get("source_doc", ""),
            "page_number": int(meta.get("page_number", 0)),
            "section_heading": meta.get("section_heading") or None,
            "distance": float(dist),
        })
    return {"results": rows}

_ALLOWED_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
}
_ALLOWED_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "exp": math.exp,
}

def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        return _ALLOWED_FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("Unsupported calculator expression")

def calculator_tool(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree))

def python_interpreter_tool(code: str) -> str:
    tree = ast.parse(code, mode="exec")
    forbidden = (
        ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith,
        ast.FunctionDef, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError("Imports, classes, functions, and context managers are not allowed.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are not allowed.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attributes are not allowed.")

    env = {"math": math}
    exec(compile(tree, "<safe-python>", "exec"), {"__builtins__": {}}, env)
    return str({k: v for k, v in env.items() if k != "math"})
