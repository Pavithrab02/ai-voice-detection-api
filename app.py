import gradio as gr

def predict(text):
    return f"You said: {text}"

demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="AI Voice Detection Demo"
)

if __name__ == "__main__":
    demo.launch()
