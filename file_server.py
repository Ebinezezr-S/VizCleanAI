# file_server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import uvicorn

app = FastAPI()
BASE = Path(__file__).parent / "deploy"

@app.get("/files")
def list_files():
    files = sorted([f.name for f in BASE.glob("*") if f.is_file()], reverse=True)
    return {"files": files}

@app.get("/files/{filename}")
def get_file(filename: str):
    fpath = BASE / filename
    if not fpath.exists() or not fpath.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(fpath), filename=filename)

if __name__ == "__main__":
    uvicorn.run("file_server:app", host="127.0.0.1", port=8502, reload=True)
