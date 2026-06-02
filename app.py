import os
import gradio as gr
from openai import OpenAI

# 🚨 GROQ CONFIGURATION: Groq ki API key aur uska official server link setup
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://groq.com"
)

def run_medical_scribe(audio_path):
    if not audio_path:
        return "Bhai, pehle microphone se audio record karein!"
    
    try:
        # 1. Speech to Text via Groq Whisper Model
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3", # Groq ka superfast whisper model
                file=audio_file
            )
        user_text = transcript.text
        
        if not user_text.strip():
            return "Audio clear nahi tha bhai. Dubara try karein!"
        
        # 2. Clinical Structure Engineering Prompt
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
        
        # 3. Request generation using Grok LLM Model (Llama 3)
        response = client.chat.completions.create(
            model="llama3-70b-8192", # Groq ka sabse powerful aur fast open-source model
            messages=[
                {"role": "system", "content": "You are a professional medical documentation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices.message.content

    except Exception as e:
        return f"System Error aaya bhai: {str(e)}\n\nKripya check karein ki Render dashboard me 'GROQ_API_KEY' sahi se daali hai ya nahi."

# Gradio UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 AI Medical Scribe Dashboard (Groq-Powered)")
    gr.Markdown("Render platform par securely chalne wala 24/7 live clinical reporting tool.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Click mic to record conversation")
            submit_btn = gr.Button("🚀 Generate Clinical Notes", variant="primary")
        with gr.Column(scale=2):
            output_box = gr.Textbox(label="📋 Generated Medical Notes", lines=20, show_copy_button=True)

    submit_btn.click(fn=run_medical_scribe, inputs=audio_input, outputs=output_box)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
