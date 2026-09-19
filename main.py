import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import CORS_ORIGINS, UPLOAD_DIR, REPORT_DIR
from init_db import init_database
from routers import auth, officer, admin, consumer

# Initialize Database and First-time Admin
init_database()

app = FastAPI(
    title="CLARO API",
    description="AI-Powered Legal Metrology Inspection & Product Intelligence Platform",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for evidence images and PDF reports
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")

# Include Routers
app.include_router(auth.router)
app.include_router(officer.router)
app.include_router(admin.router)
app.include_router(consumer.router)

@app.get("/")
def root():
    return {
        "platform": "CLARO",
        "tagline": "INSPECT • MONITOR • UNDERSTAND",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "claro-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
