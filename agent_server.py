from fastapi import FastAPI
import os

BASE_DIR = r"E:\Development_back_up_folder_2026\colab_dual_platform_unification"

app = FastAPI()

@app.get("/")
def root():
    return {"status": "agent access active"}

@app.get("/list")
def list_files():
    return {"files": os.listdir(BASE_DIR)}

@app.get("/read/{filename}")
def read_file(filename: str):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return {"error": "file not found"}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return {"content": f.read()}