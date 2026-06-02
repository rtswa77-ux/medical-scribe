import os
import gradio as gr
from openai import OpenAI

# OpenAI client setup (Render cloud service direct environment variable se API key check karega)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_medical_scribe(audio_path):
    if audio_path is None:
        return "Bhai, pehle audio record karein!"
    
    try:
        # 1. Speech to Text API Call
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        user_text = transcript.text
        
        # 2. LLM Prompt Processing
        prompt = f"""You are an expert medical scribe. Process this conversation and generate:
        1. SOAP Notes
        2. ICD-10 Codes
        3. Mental Status Examination (MSE) notes.
        
        Conversation: {user_text}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices.message.content
        
    except Exception as e:
        return f"Error aaya bhai: {str(e)}"

# Gradio Clean UI Layout Setup
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 AI Medical Scribe Dashboard (24/7 Live)")
    gr.Markdown("Render platform par securely chalne wala clinical summary tool.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Record Patient Conversation")
            submit_btn = gr.Button("🚀 Generate Medical Notes", variant="primary")
        with gr.Column(scale=2):
            output_box = gr.Textbox(label="📋 Clinical Notes (SOAP, MSE, ICD)", lines=18, show_copy_button=True)

    submit_btn.click(fn=run_medical_scribe, inputs=audio_input, outputs=output_box)

# Render server validation compliance rules setup
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
