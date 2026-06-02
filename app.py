import os
import gradio as gr
from openai import OpenAI

# 1. Grok API Configuration Client
# Render ke environment variable se 'GROK_API_KEY' automatic load hogi
client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://xai.ai"
)

def run_medical_scribe(audio_path):
    if not audio_path:
        return "Bhai, pehle microphone se audio record karein!"
    
    try:
        # 2. Speech to Text via Audio Transcription API
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        user_text = transcript.text
        
        if not user_text.strip():
            return "Audio clear nahi tha bhai, ya fir koi aawaz nahi aayi. Dubara try karein!"
        
        # 3. Clinical Structure Engineering Prompt
        prompt = f"""You are an elite medical scribe. Synthesize the following doctor-patient audio transcript into a structured clinical document. 
        Provide exactly these three sections using clear Markdown headers and bullet points:

        # 🩺 CLINICAL DOCUMENTATION REPORT
        
        ## 1. SOAP Notes
        - Subjective (Symptoms, patient history, complaints)
        - Objective (Vital signs, physical exams mentioned)
        - Assessment (Differential diagnoses, clinical impression)
        - Plan (Next steps, medications, follow-ups)

        ## 2. ICD-10 Diagnosis Codes
        - List valid ICD-10 codes with descriptive definitions relevant to the case.

        ## 3. Mental Status Examination (MSE)
        - Appearance & Behaviour
        - Speech (Coherence, rate, volume)
        - Mood & Affect
        - Cognition & Thought Content

        Transcript data to process:
        "{user_text}"
        """
        
        # 4. Request generation using Grok LLM Model
        response = client.chat.completions.create(
            model="grok-2-1212",
            messages=[
                {"role": "system", "content": "You are a professional, HIPAA-compliant medical documentation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices.message.content

    except Exception as e:
        return f"System Error aaya bhai: {str(e)}\n\nKripya check karein ki Render dashboard me 'GROK_API_KEY' sahi se daali hai ya nahi."

# 5. Gradio Functional Dashboard Layout Setup
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 AI Medical Scribe Dashboard (Grok-Powered)")
    gr.Markdown("Render cloud platform par securely chalne wala 24/7 live clinical reporting tool.")
    
    with gr.Row():
        # Left Panel Controls
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="🎤 Click mic to record conversation"
            )
            submit_btn = gr.Button("🚀 Generate Clinical Notes", variant="primary")
            
        # Right Panel Display Outputs
        with gr.Column(scale=2):
            output_box = gr.Textbox(
                label="📋 Generated Medical Notes (SOAP, ICD-10, MSE)", 
                lines=20, 
                show_copy_button=True
            )

    submit_btn.click(
        fn=run_medical_scribe, 
        inputs=audio_input, 
        outputs=output_box
    )

# 6. Web Deployment Compliance Binding Ports
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
