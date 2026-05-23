import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def generate_caption(image_path):
    """
    Loads an image, extracts visual features using a pre-trained Vision encoder,
    and generates a natural language description using a Decoder network.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None

    print("🔄 Loading pre-trained AI Model (Microsoft GIT)...")
    # Using 'microsoft/git-base' which handles both Image Processing and Text Generation
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    # 1. Open and preprocess the image (Computer Vision step)
    print("📸 Processing image features...")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # 2. Generate text tokens from visual features (NLP Decoder step)
    print("📝 Generating caption...")
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    
    # 3. Decode tokens into a human-readable string
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return caption

if __name__ == "__main__":
    # Change this to whatever sample image you place in your folder
    image_name = "sample_image.jpg" 
    
    print("=========================================")
    print("      IMAGE CAPTIONING AI SYSTEM         ")
    print("=========================================\n")
    
    # Run the model
    caption_result = generate_caption(image_name)
    
    if caption_result:
        print("\n✨ AI Generated Caption:")
        print(f"👉 \"{caption_result.capitalize()}\"")
    print("\n=========================================")