
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

image = cv2.imread("09_Object_Detection/test_image.jpg")

if image is None:
    print("Error: Image not found.")
else:
    results = model(image)

    output = results[0].plot()

    cv2.imshow("YOLO Object Detection", output)
    cv2.imwrite("09_Object_Detection/detected_image.jpg", output)

    print("Detected Objects:")
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            print(model.names[class_id], round(confidence * 100, 2), "%")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
