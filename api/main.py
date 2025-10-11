# api/main.py
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from data_pipeline.preprocess import clean_pipeline

app = FastAPI(title="vizclean API")
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    target = UPLOAD_DIR / file.filename
    contents = await file.read()
    target.write_bytes(contents)
    return JSONResponse({"filename": file.filename, "size": len(contents)})


@app.post("/clean")
async def clean(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    df, meta = clean_pipeline(str(path))
    cleaned_path = UPLOAD_DIR / f"cleaned_{filename}"
    df.to_csv(cleaned_path, index=False)
    return {"cleaned_file": str(cleaned_path), "meta": meta}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
