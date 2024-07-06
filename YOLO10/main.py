from ultralytics import YOLOv10

if __name__ == '__main__':
    # Load a model
    model = YOLOv10('yolov10x.yaml')  # build a new model from YAML
    model = YOLOv10('yolov10x.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov10n.yaml').load('yolov10n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='1_custom.yaml', epochs=500, imgsz=640,batch=4)

