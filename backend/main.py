import fastapi
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI(title="FastAPI Plex Clone")

@app.get("/", tags=["meta"])
async def root():
    return {"message": "fastapi server is running"}