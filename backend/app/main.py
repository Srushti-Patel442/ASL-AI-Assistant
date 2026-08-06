"""
File: main.py

Purpose:
Starts the ASL AI Assistant backend using FastAPI.
Defines the application's API endpoints.
http://127.0.0.1:8000/docs test backend endpoints
"""

from fastapi import FastAPI

#create FastAPI application
app=FastAPI(
    title="ASL AI Assistant API",
    version="1.0.0"
)

# GET request to the root URL (http://127.0.0.1:8000/)
# Used to check that the backend is running correctly.
@app.get("/")#
def home():
    # Return JSON data to whoever calls this endpoint
    return {
        "project":"ASL AI Assistant",
        "status":"Running",
        "message":"Welcome to the ASL AI Assistant backend"
    }