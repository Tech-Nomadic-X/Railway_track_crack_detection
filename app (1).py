import gradio as gr
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Prediction function
def detect(image):
    results = model(image)
    
    # Plot result image with boxes
    output_image = results[0].plot()
    
    return output_image

# Gradio interface
demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title="Railway Crack Detection",
    description="Upload a railway track image to detect cracks"
)

demo.launch()
