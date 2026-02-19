from transformers import BlipProcessor , BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True )
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True)

def generate_caption (img_input):
    img_input = Image.open(img_input).convert("RGB")
    inputs = processor(images=img_input, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption 