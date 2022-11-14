import torch
import sys

sys.path.insert(0, './yolov7')

device = torch.device('cpu')
model = torch.load('./yolov7/best.pt', map_location=device)['model'].float()
torch.onnx.export(model, torch.zeros((1, 3, 640, 640)), 'best_tazas.onnx', opset_version=12)