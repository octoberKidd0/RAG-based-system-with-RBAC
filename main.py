# main.py

import os
import re
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth import (
    init_database, authenticate_user, create_token,
    decode_token, revoke_token, can_upload,
    get_all_access_levels, ROLE_HIERARCHY,
)
from rag_engine import query_with_rbac
from document_loader import (
    build_vector_store, add_file_to_store,
    SUPPORTED_EXTENSIONS, get_documents_hash,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = 50
MAX_HISTORY   = 20

# ── State ─────────────────────────────────────────────────────────────────────
vectorstore = None
conversation_history: dict[str, list[dict]] = defaultdict(list)
index_lock = threading.Lock()

FOLDERS = {
    "documents/public":   "public",
    "documents/employee": "employee",
    "documents/admin":    "admin",
}


# ── Background folder watcher ─────────────────────────────────────────────────
# Runs in a background thread every 10 seconds.
# Picks up any file dropped manually into the documents folders and
# indexes it without a restart — same as the upload endpoint does.

def folder_watcher():
    global vectorstore
    indexed_files: set[str] = set()
    time.sleep(5)   # wait for server to fully start before first scan

    while True:
        for folder, access_level in FOLDERS.items():
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                full_path = os.path.join(folder, fname)
                if full_path not in indexed_files:
                    with index_lock:
                        vectorstore = add_file_to_store(
                            full_path, access_level, vectorstore
                        )
                    indexed_files.add(full_path)
        time.sleep(10)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    init_database()
    vectorstore = build_vector_store()
    threading.Thread(target=folder_watcher, daemon=True).start()
    print("✓ Server ready!")
    yield


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Chatbot with RBAC", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request models ────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str
    token: str

class TokenRequest(BaseModel):
    token: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """
    Strips path components and dangerous characters from an uploaded filename.
    Prevents path traversal attacks like "../../auth.py".
    """
    name = os.path.basename(name)
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name or "upload"


def require_auth(token: str):
    """Decodes token and returns payload, or raises HTTP 401."""
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again."
        )
    return payload


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/validate-token")
async def validate_token(request: TokenRequest):
    """
    Called by the frontend on page load to check if a saved token is still valid.
    Returns 200 if valid, 401 if expired or revoked.
    This is what allows the session to survive a page reload.
    """
    payload = decode_token(request.token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token invalid or expired.")
    return {"valid": True, "username": payload["sub"], "role": payload["role"]}


@app.post("/login")
async def login(request: LoginRequest, req: Request):
    ip   = req.client.host if req.client else None
    user = authenticate_user(request.username, request.password, ip=ip)

    if user == "locked":
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Please wait 15 minutes and try again."
        )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    conversation_history[user["username"]] = []
    token = create_token(user["username"], user["role"])

    return {
        "token":         token,
        "username":      user["username"],
        "role":          user["role"],
        "can_upload":    can_upload(user["role"]),
        # Full access level list so the UI can build the upload dropdown
        "access_levels": get_all_access_levels(),
    }


@app.post("/logout")
async def logout(request: TokenRequest):
    """
    Immediately invalidates the token server-side.
    A stolen token becomes useless the moment the user logs out.
    """
    revoke_token(request.token)
    return {"message": "Logged out successfully."}


@app.post("/chat")
async def chat(request: ChatRequest):
    payload  = require_auth(request.token)
    username = payload["sub"]
    role     = payload["role"]

    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Document store not ready yet.")

    history = conversation_history[username]
    result  = query_with_rbac(role, request.question, vectorstore, history=history)

    # result is either None (nothing found) or {"answer": ..., "sources": [...]}
    if result is None:
        answer  = (
            "I couldn't find relevant information in the documents available "
            "to your role. Try rephrasing, or check if the relevant document "
            "has been uploaded."
        )
        sources = []
    else:
        answer  = result["answer"]
        sources = result["sources"]

    # Save exchange to history for follow-up question support
    history.append({"role": "user",      "content": request.question})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY:
        conversation_history[username] = history[-MAX_HISTORY:]

    return {
        "answer":   answer,
        "sources":  sources,   # [{"source": "file.pdf", "page": 3}, ...]
        "role":     role,
        "username": username,
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    access_level: str = Form(...),
    token: str = Form(...)
):
    """Upload a PDF and rebuild the vector index."""
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload documents.")

    # Save the file to the correct folder
    folder = f"documents/{access_level}"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Rebuild the vector store
    global vectorstore
    vectorstore = build_vector_store()

    return {"message": f"✓ {file.filename} uploaded to {access_level} and index rebuilt."}


@app.get("/documents")
async def list_documents(token: str):
    """
    Returns a list of all indexed documents visible to the requesting user's role.
    Used by the Documents tab in the UI.
    """
    payload = require_auth(token)
    role    = payload["role"]

    from auth import get_allowed_levels
    allowed = set(get_allowed_levels(role))

    docs = []
    for folder, access_level in FOLDERS.items():
        if access_level not in allowed:
            continue
        if not os.path.exists(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                docs.append({
                    "filename":     fname,
                    "access_level": access_level,
                    "folder":       folder,
                })

    return {"documents": docs}


@app.post("/clear-history")
async def clear_history(request: TokenRequest):
    payload = require_auth(request.token)
    conversation_history[payload["sub"]] = []
    return {"message": "Conversation history cleared."}


@app.post("/rebuild-index")
async def rebuild_index(request: TokenRequest):
    """Admin-only full rebuild — useful after manually editing files."""
    global vectorstore
    payload = require_auth(request.token)
    if payload.get("role") not in ("admin",):
        raise HTTPException(status_code=403, detail="Admin access required.")
    with index_lock:
        vectorstore = build_vector_store()
    return {"message": "Vector store rebuilt successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)