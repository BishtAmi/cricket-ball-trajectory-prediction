from ultralytics import YOLO
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected:")
    for gpu in gpus:
        print(f" - {gpu.name}")
        tf.config.experimental.set_memory_growth(gpu, True)  
else:
    print("No GPU detected. Training will proceed on the CPU.")

model = YOLO("yolov8s.pt")
results = model.train(data="data.yaml", epochs=5)
