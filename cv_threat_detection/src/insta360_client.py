import cv2

print("Scanning 0–15 for cameras...")

for i in range(16):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    opened = cap.isOpened()
    print(f"Index {i}: {'OPEN' if opened else '---'}")
    if opened:
        ret, frame = cap.read()
        print("   Read frame:", ret, " | Frame shape:", None if not ret else frame.shape)
    cap.release()

print("Done.")
