import torch
from tqdm import tqdm
import json

from openpose.src.model import handpose_model

gpu_available = torch.cuda.is_available()
if gpu_available:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

model = handpose_model()

size = {}
for i in tqdm(range(10, 1000)):
    data = torch.randn(1, 3, i, i)
    if torch.cuda.is_available():
        data = data.to(device)
    size[i] = model(data).size(2)

with open('hand_model_output_size.json') as f:
    json.dump(size, f)
