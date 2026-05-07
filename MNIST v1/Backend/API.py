from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import base64
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    image: str

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = MLP()
model.load_state_dict(torch.load("mnist_mlp.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict")
async def predict(data: ImageInput):
    try:
        base64_str = data.image.split(',')[1] if ',' in data.image else data.image
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert('L') 

        invert_img = img.point(lambda p: p > 20 and 255) 
        bbox = invert_img.getbbox()
        
        if bbox:
            img = img.crop(bbox) 
            
            width, height = img.size
            max_dim = max(width, height)
            padding = int(max_dim * 0.4) 
            new_size = max_dim + padding
            
            container = Image.new('L', (new_size, new_size), 0)
            container.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
            img = container

        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        
        img_tensor = torch.tensor(img_array).reshape(1, 784)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
            
        return {
            'prediction': prediction,
            'confidence': round(confidence * 100, 1),
            'probabilities': [round(p * 100, 1) for p in probabilities.tolist()]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)