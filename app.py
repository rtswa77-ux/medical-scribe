import os
import gradio as gr
from groq import Groq

def run_medical_scribe(audio_filepath):
    # --- Key Handling ---
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "## ❌ Configuration Error\n\n"
            "The `GROQ_API_KEY` environment variable is not set.\n\n"
            "Please add it in your Render.com dashboard under **Environment > Environment Variables**."
        )

    if audio_filepath is None:
        return "## ⚠️ No Audio Detected\n\nPlease record a conversation using the microphone before generating notes."

    try:
        client = Groq(api_key=api_key)

        # --- Step 1: Speech-to-Text Transcription ---
        with open(audio_filepath, "rb") as audio_file:
            transcription_response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )

        transcript = transcription_response if isinstance(transcription_response, str) else transcription_response.text

        if not transcript or transcript.strip() == "":
            return "## ⚠️ Empty Transcript\n\nNo speech was detected in the recording. Please try again with a clearer recording."

        # --- Step 2: Clinical Note Generation ---
        system_prompt = """You are an expert AI Medical Scribe and Clinical Documentation Specialist.
Your task is to analyze a doctor-patient conversation transcript and produce a structured clinical documentation report.

You MUST format your entire response in clean Markdown, exactly following this structure:

# CLINICAL DOCUMENTATION REPORT

---

## 1. SOAP Notes

**Subjective:**
(Patient's chief complaint, history of present illness, symptoms as described by the patient)

**Objective:**
(Observable/measurable findings: vitals, physical exam findings, lab results if mentioned)

**Assessment:**
(Clinical impression, diagnosis or differential diagnoses)

**Plan:**
(Treatment plan, medications, referrals, follow-up instructions)

---

## 2. ICD-10 Diagnosis Codes

| ICD-10 Code | Description | Justification |
|-------------|-------------|---------------|
| (code)      | (name)      | (why it applies based on transcript) |

(List all relevant codes. If no clear diagnosis, provide the most likely based on symptoms.)

---

## 3. Mental Status Examination (MSE)

**Appearance:** (Describe patient's presentation if mentioned or inferable)

**Speech:** (Rate, rhythm, volume, tone if mentioned or inferable)

**Mood & Affect:** (Patient's expressed mood and observed affect)

**Thought Process:** (Logical, coherent, tangential, etc.)

**Insight & Judgment:** (Patient's awareness of condition and decision-making)

---

Be thorough, professional, and clinically accurate. If certain information is not present in the transcript, note it as 'Not documented' rather than fabricating details."""

        user_message = f"""Please analyze the following doctor-patient conversation transcript and generate the complete clinical documentation report:

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        clinical_notes = completion.choices[0].message.content
        return clinical_notes

    except Exception as e:
        error_message = str(e)
        return (
            f"## ❌ Processing Error\n\n"
            f"An error occurred while processing your request:\n\n"
            f"```\n{error_message}\n```\n\n"
            f"Please check your API key, ensure your audio was recorded correctly, and try again."
        )


# --- Gradio UI ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    title="AI Medical Scribe",
    css="""
        .header-text { text-align: center; margin-bottom: 10px; }
        .generate-btn { min-height: 56px; font-size: 16px !important; font-weight: 600 !important; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown(
        """
        # 🩺 AI Medical Scribe
        ### Powered by Groq · Whisper Large V3 · LLaMA 3.3 70B
        Record a doctor-patient conversation and instantly generate structured clinical notes.
        """,
        elem_classes=["header-text"]
    )

    gr.Markdown("---")

    with gr.Row(equal_height=True):

        # --- Left Column: Input ---
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Record Consultation")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Click mic to record conversation",
                show_download_button=False,
            )
            gr.Markdown(
                "<small>💡 Speak clearly. Record the full doctor-patient exchange for best results.</small>"
            )
            generate_btn = gr.Button(
                "⚕️ Generate Clinical Notes",
                variant="primary",
                elem_classes=["generate-btn"]
            )
            gr.Markdown(
                """
                ---
                **How it works:**
                1. 🎙️ Record the consultation
                2. ⚕️ Click Generate
                3. 📋 Review & copy notes

                **Generates:**
                - SOAP Notes
                - ICD-10 Codes
                - Mental Status Exam
                """
            )

        # --- Right Column: Output ---
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Generated Clinical Notes")
            notes_output = gr.Textbox(
                label="Generated Medical Notes",
                lines=30,
                show_copy_button=True,
                placeholder=(
                    "Clinical notes will appear here after you record a conversation "
                    "and click 'Generate Clinical Notes'...\n\n"
                    "Output includes:\n"
                    "  • SOAP Notes\n"
                    "  • ICD-10 Diagnosis Codes\n"
                    "  • Mental Status Examination (MSE)"
                ),
                interactive=True,
            )

    # --- Event Binding ---
    generate_btn.click(
        fn=run_medical_scribe,
        inputs=[audio_input],
        outputs=[notes_output],
        show_progress="full",
    )

    gr.Markdown(
        "<center><small>⚠️ For clinical decision support only. Always verify AI-generated notes before use in medical records.</small></center>"
    )


# --- Render.com Free Tier Port Binding ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port
