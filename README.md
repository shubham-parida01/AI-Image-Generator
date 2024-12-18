# AI-Image-Generator

## Project Overview
This is a simple tutorial project created to explore and experiment with **Stable Diffusion** models for generating images from textual prompts. The project was done purely because I find topics like these interesting and wanted to dive into text-to-image generation.

The code was run in **Google Colab** using the **T4 runtime** type, taking advantage of its GPU resources for faster processing.

## Requirements
- Python 3.x
- Google Colab (Recommended for GPU acceleration)
- PyTorch (version >= 1.9.0)
- Hugging Face Diffusers library
- Matplotlib

## Installation Instructions
1. Install the required libraries in your Colab notebook:
   ```python
   !pip install torch torchvision
   !pip install diffusers
   !pip install matplotlib
   ```

2. Ensure you're using a T4 runtime on Google Colab by going to the menu **Runtime** > **Change runtime type**, and selecting **GPU**.

## Code Walkthrough

### Step 1: Import Required Libraries
```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
```
- **torch**: Used for GPU acceleration via PyTorch.
- **diffusers**: The Hugging Face library for text-to-image generation.
- **matplotlib**: To display the generated image.

### Step 2: Clear CUDA Cache
```python
torch.cuda.empty_cache()
```
This ensures that any previous cache is cleared from the GPU, allowing for better memory management.

### Step 3: Load the Pretrained Model
```python
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
```
- **StableDiffusionPipeline**: Loads the Stable Diffusion model.
- **DPMSolverMultistepScheduler**: A specific scheduler used to improve the generation process.
- **torch.float16**: Using 16-bit precision for faster performance.

### Step 4: Generate an Image
```python
prompt = "a man with a flower"
image = pipe(prompt, width=1000, height=1000).images[0]
```
- **prompt**: The input text describing the image you want to generate.
- The generated image is stored in the `image` variable.

### Step 5: Display the Image
```python
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
```
- The image is displayed using **matplotlib**, with the axis turned off for a cleaner presentation.

## Running the Code
1. Copy and paste the code into a Colab notebook.
2. Make sure your Colab runtime is set to GPU (T4 recommended).
3. Run the cells and see the generated image based on the input prompt!

## Notes
- This project is a tutorial I created purely because I find these topics interesting and wanted to try text-to-image generation.

## Future Improvements
- Experiment with different prompts to generate a variety of images.
- Try using other image generation models from Hugging Face.
- Implement additional post-processing techniques for image refinement.

## Acknowledgments
- **Hugging Face** for the diffusers library and pretrained models.
- **Google Colab** for providing free GPU access.

## Link
- Link for the [**Google Colab**](https://colab.research.google.com/drive/1O1qzBbj-BqVZd0vR6cAmggfw2kyzecm9?usp=sharing)

## License
This project is for educational and personal use. Please adhere to the terms and conditions provided by the libraries and models used.
