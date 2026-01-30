from ultralytics import YOLO

# Возомжно нужно будет не 608х608, а 640х640
model = YOLO("yolov10n.pt")

results = model.train(data='data.yaml',
                      optimizer='Adam',
                      lr0=0.001,
                      lrf=1.0,
                      momentum=0.9,
                      weight_decay=0.0005,
                      epochs=50,
                      batch=4,
                    #   shuffle=True,  #?
                      workers=4,
                      plots=True,
                      exist_ok=True
                      )
# no checkpoint

model.export(format="pt")
