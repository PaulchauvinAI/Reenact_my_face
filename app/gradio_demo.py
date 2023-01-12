from make_video import gradio_funct
import gradio as gr

demo = gr.Interface(
    fn=gradio_funct,
    inputs=[gr.Image(source="webcam"), gr.Slider(0, 1)],
    outputs=["playable_video"],
)
demo.launch(share=True)
