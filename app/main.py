from fastapi import FastAPI, Request, BackgroundTasks
from dotenv import load_dotenv
import os

# Import our custom Twilio webhook handler
from app.services.twilio_sms import handle_incoming_whatsapp

# Load environment variables from the .env file located in the root directory
load_dotenv()

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="FarmerChat API",
    description="Backend service for AI-powered agricultural advisory using WhatsApp.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """
    A simple health check endpoint to verify the server is running successfully.
    """
    return {
        "status": "active",
        "service": "FarmerChat",
        "message": "The API server is up and running."
    }

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request,background_tasks:BackgroundTasks):
    """
    This is the endpoint exposed to the internet via ngrok.
    Twilio will send HTTP POST requests here whenever a user sends a WhatsApp message.
    """
    # Delegate the request parsing and response generation to our dedicated service function
    return await handle_incoming_whatsapp(request,background_tasks)

if __name__ == "__main__":
    import uvicorn
    # Run the server locally on port 8000 when the script is executed directly
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)