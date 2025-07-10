import npclip
import numpy as np
from PIL import Image

model, preprocess = npclip.load("ViT-B/32")

image = preprocess(Image.open("data/CLIP.png"))[np.newaxis, :, :, :]

text = npclip.tokenize(["a diagram", "a dog", "a cat"])

image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits_per_image, logits_per_text = model(image, text)
probs = npclip.softmax(logits_per_image, axis=-1)

print("Label probs:", probs)  # prints: [[0.99279356 0.00421067 0.00299573]]
