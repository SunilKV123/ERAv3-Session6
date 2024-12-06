import torch
from datetime import datetime

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_model(model, accuracy, path="models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_model_{accuracy:.2f}acc_{timestamp}.pth"
    torch.save(model.state_dict(), f"{path}/{filename}")
    return filename

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model 