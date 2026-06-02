import os
import gradio as gr
from groq import Groq

def run_medical_scribe(audio_path):
    if not audio_path:
        return "Bhai, pehle microphone se audio record karein!"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: Render dashboard me 'GROQ_API_KEY' nahi mili bhai."

    try:
        client = Groq(api_key=api_key)

        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file
            )

        user_text = transcript.text

        if not user_text.strip():
            return "Audio clear nahi tha bhai. Dubara try karein!"

        prompt = f"""You are an elite medical scribe. Synthesize the following doctor-patient audio transcript into a structured clinical document.
Provide exactly these three sections using clear Markdown headers:

# CLINICAL DOCUMENTATION REPORT

## 1. SOAP Notes
- Subjective (Symptoms, complaints)
- Objective (Vital signs, exams)
- Assessment (Diagnoses)
- Plan (Medications, follow-ups)

## 2. ICD-10 Diagnosis Codes
- List valid ICD-10 codes with descriptive definitions.

## 3. Mental Status Examination (MSE)
- Appearance, Speech, Mood & Affect.

Transcript data to process: "{user_text}" """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional medical documentation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"System Error aaya bhai: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# AI Medical Scribe Dashboard")
    gr.Markdown("Render platform par securely chalne wala 24/7 live clinical reporting tool.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Click mic to record conversation")
            submit_btn = gr.Button("Generate Clinical Notes", variant="primary")

        with gr.Column(scale=2):
            output_box = gr.Textbox(label="Clinical Notes (SOAP, ICD-10, MSE)", lines=20, show_copy_button=True)

    submit_btn.click(fn=run_medical_scribe, inputs=audio_input, outputs=output_box)

port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
