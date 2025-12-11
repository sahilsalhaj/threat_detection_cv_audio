from ultralytics import YOLO

model = model = YOLO("cv_threat_detection/models/best.pt")

print("\n=== MODEL INFO ===")
model.info()

print("\n=== MODEL YAML ===")
print(model.model.yaml)

print("\n=== TRAINING ARGS (if available) ===")
if hasattr(model, "args"):
    print(model.args)

