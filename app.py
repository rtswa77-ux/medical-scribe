import os
import gradio as gr
from groq import Groq

def run_medical_scribe(audio_filepath):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "## ❌ Configuration Error\n\n"
            "`GROQ_API_KEY` environment variable is not set.\n\n"
            "Please add it in Render dashboard → Environment → Environment Variables."
        )

    if audio_filepath is None:
        return "## ⚠️ No Audio\n\nPlease record a conversation first."

    try:
        client = Groq(api_key=api_key)

        # Step 1: Transcription
        with open(audio_filepath, "rb") as audio_file:
            transcription_response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )

        transcript = transcription_response if isinstance(transcription_response, str) else transcription_response.text

        if not transcript or transcript.strip() == "":
            return "## ⚠️ Empty Transcript\n\nNo speech detected. Please try again."

        # Step 2: Clinical Notes
        system_prompt = """You are an expert AI Medical Scribe and Clinical Documentation Specialist.
Analyze the doctor-patient conversation and produce structured clinical documentation in Markdown.

# CLINICAL DOCUMENTATION REPORT

---

## 1. SOAP Notes

**Subjective:**
(Chief complaint, history, symptoms)

**Objective:**
(Vitals, exam findings, labs)

**Assessment:**
(Diagnosis or differentials)

**Plan:**
(Treatment, medications, follow-up)

---

## 2. ICD-10 Diagnosis Codes

| ICD-10 Code | Description | Justification |
|-------------|-------------|---------------|
| (code) | (name) | (reason) |

---

## 3. Mental Status Examination (MSE)

**Appearance:** 
**Speech:** 
**Mood & Affect:** 
**Thought Process:** 
**Insight & Judgment:** 

---

If info is missing, write 'Not documented'."""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript:\n{transcript}"}
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"## ❌ Error\n\n```\n{str(e)}\n```"


# Gradio UI
with gr.Blocks(title="AI Medical Scribe") as demo:

    gr.Markdown("# 🩺 AI Medical Scribe\n### Groq · Whisper Large V3 · LLaMA 3.3 70B")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Record Consultation")
            audio_input = gr.Audio(
                source="microphone",
                type="filepath",
                label="Click mic to record conversation"
            )
            generate_btn = gr.Button("⚕️ Generate Clinical Notes", variant="primary")
            gr.Markdown("**Generates:** SOAP Notes · ICD-10 Codes · MSE")

        with gr.Column(scale=2):
            gr.Markdown("### 📋 Generated Clinical Notes")
            notes_output = gr.Textbox(
                label="Generated Medical Notes",
                lines=30,
                show_copy_button=True,
                placeholder="Clinical notes will appear here..."
            )

    generate_btn.click(
        fn=run_medical_scribe,
        inputs=[audio_input],
        outputs=[notes_output]
    )

    gr.Markdown("⚠️ *For clinical decision support only. Always verify before use in medical records.*")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)

    
