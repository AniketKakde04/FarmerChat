import os
import requests
from fastapi import Request, Response, BackgroundTasks
from twilio.rest import Client
from dotenv import load_dotenv

# Import our LangGraph workflow and our AI functions
from app.graph.workflow import app_graph
from app.services.sarvam_api import transcribe_audio
from app.services.vision_api import analyze_crop_image # ADDED THIS IMPORT

load_dotenv()

def background_process_and_reply(sender: str, twilio_number: str, user_message: str, num_media: int, form_data: dict):
    """This function runs in the background so Twilio doesn't time out."""
    
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    
    # 1. Check for Media (Audio or Image)
    if num_media > 0:
        media_type = form_data.get('MediaContentType0', '')
        media_url = form_data.get('MediaUrl0', '')
        
        # --- AUDIO HANDLING ---
        if media_type.startswith('audio/'):
            print(f"--- [TWILIO] Incoming Voice Note Detected ---")
            audio_response = requests.get(media_url, auth=(account_sid, auth_token))
            
            if audio_response.status_code == 200:
                temp_file = "temp_voice_note.ogg"
                with open(temp_file, "wb") as f:
                    f.write(audio_response.content)
                
                transcribed_text = transcribe_audio(temp_file)
                user_message = transcribed_text if transcribed_text else "Hello"
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # --- IMAGE HANDLING (ADDED THIS BLOCK) ---
        elif media_type.startswith('image/'):
            print(f"--- [TWILIO] Incoming Image Detected ---")
            image_response = requests.get(media_url, auth=(account_sid, auth_token))
            
            if image_response.status_code == 200:
                temp_img = "temp_crop_image.jpg"
                with open(temp_img, "wb") as f:
                    f.write(image_response.content)
                
                # Run the Keras Vision Model
                disease_diagnosis = analyze_crop_image(temp_img)
                
                # Create a prompt for LangGraph based on the AI's diagnosis
                if disease_diagnosis.lower() in ["fresh cotton leaf", "fresh cotton plant"]:
                    user_message = f"I am a farmer. I just uploaded a photo of my crop. The AI vision system diagnosed it as '{disease_diagnosis}'. Please tell me my crop looks healthy and give me a brief general tip for maintaining healthy cotton."
                else:
                    user_message = f"I am a farmer. I just uploaded a photo of my sick crop. The AI vision system diagnosed it as: '{disease_diagnosis}'. Search your knowledge base and give me detailed treatment advice in Marathi."
                
                if os.path.exists(temp_img):
                    os.remove(temp_img)

    print("\n--- Processing WhatsApp Message ---")
    print(f"From: {sender}")
    print(f"Message: {user_message}")
    print("---------------------------------\n")

    # 2. Pass to LangGraph
    initial_state = {"user_query": user_message}
    final_state = app_graph.invoke(initial_state)
    bot_reply = final_state.get("final_answer", "Error: No answer generated.")

    # 3. Send the final answer back via Twilio REST Client
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_=twilio_number,
            body=bot_reply,
            to=sender
        )
        print(f"--- [TWILIO] Success! Reply sent to WhatsApp (SID: {message.sid}) ---")
    except Exception as e:
        print(f"--- [TWILIO] Error sending message: {e} ---")


async def handle_incoming_whatsapp(request: Request, background_tasks: BackgroundTasks):
    """Instantly accepts the request and pushes the heavy lifting to the background."""
    form_data = await request.form()
    
    sender = form_data.get('From', '')
    twilio_number = form_data.get('To', '')
    num_media = int(form_data.get('NumMedia', 0))
    user_message = form_data.get('Body', '').strip()
    
    # Send the heavy processing to the background worker
    background_tasks.add_task(
        background_process_and_reply, 
        sender, 
        twilio_number, 
        user_message, 
        num_media, 
        dict(form_data)
    )
    
    # Return a blank 200 OK instantly
    return Response(content="<Response></Response>", media_type="application/xml")