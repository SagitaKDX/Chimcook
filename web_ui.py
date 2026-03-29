import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Try to import RAG pipeline, gracefully handle if unavailable for UI testing
try:
    from core.rag import RAGPipeline
except ImportError:
    RAGPipeline = None

app = Flask(__name__)
DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

import stat
import pwd

# Application state
rag_instance = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files', methods=['GET'])
def list_files():
    try:
        files = []
        for f in DOCS_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in ('.txt', '.md', '.markdown'):
                files.append(f.name)
        return jsonify({"files": files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        # Prevent directory traversal mapping
        clean_name = os.path.basename(filename)
        path = DOCS_DIR / clean_name
        if path.exists() and path.is_file():
            path.unlink()
            return jsonify({"message": f"Deleted {clean_name}"}), 200
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_db_status():
    faiss_file = Path("data/faiss_index/index.faiss")
    if faiss_file.exists():
        size_kb = faiss_file.stat().st_size / 1024
        return jsonify({
            "trained": True, 
            "message": f"Active (Index size: {size_kb:.1f} KB)"
        }), 200
    return jsonify({
        "trained": False, 
        "message": "Empty (Needs Training)"
    }), 200

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved = 0
    for file in files:
        if file.filename.lower().endswith(('.txt', '.md', '.markdown')):
            path = DOCS_DIR / file.filename
            file.save(path)
            saved += 1
            
    return jsonify({"message": f"Successfully saved {saved} files."}), 200

@app.route('/ingest', methods=['POST'])
def ingest():
    global rag_instance
    if not RAGPipeline:
        return jsonify({"error": "RAGPipeline could not be imported. Ensure dependencies are installed."}), 500
        
    try:
        if not rag_instance:
            rag_instance = RAGPipeline()
            
        rag_instance.ingest_documents()
        return jsonify({"message": "Successfully ingested files and trained FAISS index."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    import sys
    
    docs_stat = DOCS_DIR.stat() if DOCS_DIR.exists() else None
    
    debug_data = {
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "docs_dir": {
            "path": str(DOCS_DIR.absolute()),
            "exists": DOCS_DIR.exists(),
            "owner_uid": docs_stat.st_uid if docs_stat else None,
            "permissions": oct(docs_stat.st_mode)[-4:] if docs_stat else None
        },
        "files_in_docs": [],
    }
    
    if DOCS_DIR.exists():
        for f in DOCS_DIR.iterdir():
            debug_data["files_in_docs"].append({
                "name": f.name,
                "is_file": f.is_file(),
                "size": f.stat().st_size if f.is_file() else 0
            })
            
    return jsonify(debug_data), 200

if __name__ == '__main__':
    print("Starting Chimcook RAG Debug Portal...")
    print("Serving on http://localhost:5000")
    # Bind to 0.0.0.0 to allow network access if needed
    app.run(host='0.0.0.0', port=5000, debug=True)
