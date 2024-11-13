import zipfile
import os
import yaml
from ultralytics import YOLO
import subprocess


def create_yaml():
    data = {
        'nc': 2,
        'test': './test/images',
        'train': './train/images',
        'val': './test/images'
    }
    with open('config.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def train_model():
    model = YOLO("yolo11x.pt")
    results = model.train(data="config.yaml", epochs=100, imgsz=640, device=0, batch=16)
    print("Обучение завершено.")

if __name__ == "__main__":
    os.system('gdown 1Co3RN2m0HHLa5debwaOciqhTymwuBXB0')

    zip_file_path = "./train_cut2.zip"
    extract_dir = "./"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_file_path)
    create_yaml()
    train_model()
