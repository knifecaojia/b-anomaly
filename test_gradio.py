import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"f:\Bear\apple")

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Hello Test")
    btn = gr.Button("Click")
    out = gr.Textbox()
    btn.click(lambda: "Hello", outputs=[out])

demo.launch(server_name="127.0.0.1", server_port=7862)
