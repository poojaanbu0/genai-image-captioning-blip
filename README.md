## Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Automated image captioning involves generating descriptive text for visual content, an essential capability for applications in accessibility, multimedia retrieval, and automated content creation. The challenge is to produce accurate and meaningful captions using pre-trained models while ensuring ease of use for end users. This project leverages the BLIP model to address these challenges, with a Gradio-powered interface for user interaction and evaluation.

### DESIGN STEPS:
### Step 1: Set Up the Environment
- Install Required Libraries
- Ensure that the GPU runtime is enabled in Colab for faster processing.

### Step 2: Load the BLIP Model
- Load Pre-Trained BLIP Model:
    Use Hugging Face's `transformers` library to load the BLIP image captioning model.
- Preload the processor and model to handle image inputs and generate captions efficiently.

### Step 3: Define the Image Captioning Function
- Create a function to process the image input and generate a caption using the BLIP model.

### Step 4: Build the Gradio Interface
- Use Gradio to create a user-friendly interface for uploading images and displaying captions.
- Include an image uploader and a textbox for the generated caption.

  
- Execute the code and launch the Gradio app.

### PROGRAM:
```python
!pip install torch torchvision transformers gradio

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr

# Load the BLIP model
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# Define the captioning function
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Create the Gradio Interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## BLIP Image Captioning")

        input_image = gr.Image(type="pil", label="Upload an Image")
        output_caption = gr.Textbox(label="Generated Caption")
        gr.Button("Generate Caption").click(fn=generate_caption, inputs=input_image, outputs=output_caption)

        demo.launch()

if __name__ == "__main__":
    main()

```

### OUTPUT:
![image](https://github.com/user-attachments/assets/e4e822a0-dbfb-4ab9-897c-75c89ca38171)

### RESULT:
Successfully developed a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.
