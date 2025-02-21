from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
import tempfile  # ✅ Import tempfile to create temp files

# ✅ Load the fastest model on CPU
model_name = "EleutherAI/pythia-70m"  # Fastest model for code review
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")  # Force CPU mode

# ✅ Function to review Python code with debug logs
def review_code(code_snippet):
    print("✅ Received Code:", code_snippet)  # Debugging log
    
    # Process input
    inputs = tokenizer(code_snippet, return_tensors="pt").to("cpu")  # Move to CPU
    outputs = model.generate(
        **inputs,
        max_length=50,  # ✅ Reduced token generation to prevent hallucinations
        do_sample=False,
        num_beams=3,
        repetition_penalty=2.0  # ✅ Penalizes repetitive text generation
    )

    # Check if the model generated output
    if outputs is None:
        print("❌ Model did not generate output!")  # Debugging log
        return "Error: Model did not generate output."

    reviewed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("✅ Generated Code:", reviewed_code)  # Debugging log

    # ✅ Write reviewed code to a temporary file for download
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(temp_file_path, "w") as temp_file:
        temp_file.write(reviewed_code)

    return reviewed_code, temp_file_path  # ✅ Return reviewed code & file path

# ✅ Handle user input and return reviewed code
def check_code(input_code):
    reviewed_code, file_path = review_code(input_code)
    return input_code, reviewed_code, file_path  # ✅ Correctly return file path

# ✅ Gradio UI with Side-by-Side Comparison & Fixed Download Option
interface = gr.Interface(
    fn=check_code,
    inputs=gr.Textbox(label="Enter Python Code"),
    outputs=[
        gr.Textbox(label="Original Code", interactive=False),  # Left side
        gr.Textbox(label="Reviewed Code", interactive=False),  # Right side
        gr.File(label="Download Reviewed Code")  # ✅ Fixed Download Button
    ],
    title="🚀 AI Code Reviewer",
    description="📌 Enter Python code and get a reviewed version. Download the reviewed code as a file.",
    allow_flagging="never"
)

# ✅ Launch app (Fixes font issues and removes `share=True`)
interface.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
