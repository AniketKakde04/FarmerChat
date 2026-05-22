import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load environment variables to access the SARVAM_API_KEY
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# Initialize the official Sarvam client
# It will use the key if available, otherwise stay None to prevent crash on startup without a key
client = SarvamAI(api_subscription_key=SARVAM_API_KEY) if SARVAM_API_KEY and SARVAM_API_KEY != "your_sarvam_api_key_here" else None

def generate_agri_response(user_message: str, context_data: str = None, language: str = "Marathi") -> str:
    """
    Uses the Sarvam-30b model to generate native agricultural advice.
    """
    if not client:
        return "Warning: SARVAM_API_KEY is not configured."

    # SIMPLIFIED PROMPT: Removed negative constraints that trigger overthinking
    system_prompt = (
        f"You are FarmerChat, an expert agricultural advisor helping women farmers in Maharashtra. "
        f"Respond directly to the user in fluent {language}. Keep your answers warm, short, and practical."
    )
    if context_data:
        system_prompt += f"\n\nCRITICAL RULE: Base your advice ONLY on the following official data. If the answer is not in this data, say you don't know.\n\nOFFICIAL DATA:\n{context_data}"

    try:
        response = client.chat.completions(
            model="sarvam-m",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,
            top_p=1,
            max_tokens=2000  # INCREASED: Give it plenty of room to finish generating
        )
        
        print("\n--- DEBUG: RAW SARVAM RESPONSE ---")
        try:
            print(response)
        except Exception:
            try:
                print(str(response).encode('ascii', errors='replace').decode('ascii'))
            except Exception as inner_e:
                print(f"Could not print response: {inner_e}")
        print("----------------------------------\n")

        # Safely extract the message object
        message_obj = response.choices[0].message
        content = message_obj.content

        # If content is still blank, let's see if it got stuck in the 'reasoning' phase
        if content is None:
            reasoning = getattr(message_obj, 'reasoning_content', None)
            if reasoning:
                print(f"\n--- AI GOT STUCK THINKING ---")
                try:
                    print(reasoning)
                except Exception:
                    try:
                        print(str(reasoning).encode('ascii', errors='replace').decode('ascii'))
                    except Exception as inner_e:
                        print(f"Could not print reasoning: {inner_e}")
                print("-----------------------------\n")
                return "Error: AI spent too long thinking. Please ask a simpler question."
            return "Error: Sarvam AI returned a blank response. Check terminal for debug info."

        # Strip out any <think>...</think> blocks from content if present (common with sarvam-m)
        if content:
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
        return content
        
    except Exception as e:
        import traceback
        print(f"Error calling Sarvam AI: {e}")
        traceback.print_exc()
        return "क्षमस्व, मला आता तांत्रिक अडचणी येत आहेत. (Sorry, I am facing technical difficulties right now.)"


def transcribe_audio(file_path: str) -> str:
    """
    Sends the downloaded audio file to Sarvam's Speech-to-Text endpoint using the official SDK.
    """
    try:
        print(f"--- [AUDIO] Transcribing file: {file_path} ---")
        with open(file_path, "rb") as audio_file:
            # Using the official Sarvam SDK as per documentation
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3",
                mode="transcribe"
            )
            
            # The SDK returns an object, we extract the transcript text
            transcript = getattr(response, 'transcript', str(response))
            
            print(f"--- [AUDIO] Transcribed Text: {transcript} ---")
            return transcript
            
    except Exception as e:
        print(f"--- [AUDIO] Exception during transcription: {e} ---")
        return ""

if __name__ == "__main__":
    test_query = "What is the best fertilizer for cotton during the flowering stage?"
    print(f"User Query: {test_query}")
    print("\nFarmerChat AI Response:")
    response_text = generate_agri_response(test_query, language="Marathi")
    try:
        print(response_text)
    except UnicodeEncodeError:
        print(response_text.encode('ascii', errors='replace').decode('ascii'))