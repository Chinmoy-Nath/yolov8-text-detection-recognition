import cv2
import torch
import easyocr
import os
import numpy as np
from ultralytics import YOLO

# modfy the model path
model_classifier = YOLO("D:/Internship/models/DL models/card classification model.pt") # 1
model_orient = YOLO("D:/Internship/models/DL models/Card Rotation model.pt") # 2
model_text = YOLO("D:/Internship/models/DL models/Text detection model.pt") # 3
model_card_detector = YOLO("D:/Internship/models/DL models/Card detector.pt") # 4
reader = easyocr.Reader(['en'])

# def correct_orientation(image_path, predicted_class):
   
#     image = image_path

#     if image is None:
#         print("Error: Unable to read the image.")
#         return None

#     # Updated Rotation Mapping
#     rotation_map = {1: 180, 2: 270, 3: 90}

#     if predicted_class in rotation_map:
#         angle = rotation_map[predicted_class]

#         # Use cv2.rotate for 180° rotation (more efficient)
#         if angle == 180:
#             rotated = cv2.rotate(image, cv2.ROTATE_180)
#         elif angle == 90:
#             rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#         elif angle == 270:
#             rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         else:
#             rotated = image

#         return rotated

#     return image

# def crop_segment(image_path, model_path):

#     model = model_path

#     image = image_path
#     if image is None:
#         print("Error: Unable to read the image.")
#         return None

#     # Run YOLOv8 inference
#     results = model(image)

#     # Extract bounding box of the first detected object
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
#             cropped_image = image[y1:y2, x1:x2]
#             return cropped_image

#     print("No object detected!")
#     return None

# def get_details(cropped_img):
#     results = model_text(cropped_img)
#     data = {
#         0: "add",
#         1: "blood_group",
#         2: "dl_no",
#         3: "dob",
#         4: "name",
#         5: "relation_with",
#         6: "rto",
#         7: "state",
#         8: "vehicle_type"
#     }
#     output = {key: "" for key in data.values()}

#     # Loop through detected bounding boxes
#     for result in results[0].boxes:
#         x_min, y_min, x_max, y_max = map(int, result.xyxy[0])  # Extract bbox coordinates
#         class_index = int(result.cls[0])  # Extract predicted class index
#         class_name = data.get(class_index, "Unknown")  # Get class name

#         if class_name != "Unknown":
#             # Crop detected text region
#             text_region = cropped_img[y_min:y_max, x_min:x_max]

#             # Apply OCR on the cropped region
#             detected_text = reader.readtext(text_region, detail=0)

#             # Store text for the corresponding class
#             output[class_name] = " ".join(detected_text) if detected_text else ""

#     # Print the extracted information in the required format
#     print("\nExtracted Information:")
#     for key, value in output.items():
#         print(f"{key}: {value}")

# img = input("enter the image path (wihtout "" and convert \ to /) : ")

# image = cv2.imread(img)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower = np.array([0, 0, 80], dtype=np.uint8)
# upper = np.array([180, 255, 255], dtype=np.uint8)
# mask = cv2.inRange(hsv, lower, upper)
# filtered_img = cv2.bitwise_and(image, image, mask=mask)

# results = model_classifier(filtered_img)
# for r in results:
#   if (r.probs.top1 == 0):
#     print("This is DL")
#   else:
#     print("Please upload appropriate image")

# result = model_orient(filtered_img)
# predicted_class = result[0].probs.top1

# rotated_image = correct_orientation(filtered_img, predicted_class)
# cropped_img = crop_segment(rotated_image, model_card_detector)
# get_details(cropped_img)

def correct_orientation(image, predicted_class):
    """
    Rotate the input image based on the predicted class (orientation).
    """
    if image is None:
        return None
    # Map predicted class to actual rotation angle
    rotation_map = {1: 180, 2: 270, 3: 90}
    if predicted_class in rotation_map:
        angle = rotation_map[predicted_class]
        if angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = image
        return rotated
    # If no rotation needed
    return image


def crop_segment(image, model):
    """
    Runs YOLOv8 inference on the image, returns cropped region of first detected object.
    """
    if image is None:
        return None
    results = model(image)
    # Parse bboxes and crop image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image[y1:y2, x1:x2]
            return cropped_image  # Only first detected object
    return None

def get_details(cropped_img):
    """
    Detects text regions and applies OCR; returns dict of field -> detected text.
    """
    results = model_text(cropped_img)
    # Map from class index to field name
    data = {
        0: "add",
        1: "blood_group",
        2: "dl_no",
        3: "dob",
        4: "name",
        5: "relation_with",
        6: "rto",
        7: "state",
        8: "vehicle_type"
    }
    output = {key: "" for key in data.values()}
    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
        class_index = int(result.cls[0])
        class_name = data.get(class_index, "Unknown")
        if class_name != "Unknown":
            # Crop detected text region for OCR
            text_region = cropped_img[y_min:y_max, x_min:x_max]
            detected_text = reader.readtext(text_region, detail=0)
            output[class_name] = " ".join(detected_text) if detected_text else ""
    return output

def process_image(image_path_or_array):
    """
    The unified function to call for full processing.
    Accepts either image path (str) or numpy array (for Gradio).
    Returns dict of fields extracted from the document.
    """
    # If input is path, read image
    image = cv2.imread(image_path_or_array) if isinstance(image_path_or_array, str) else image_path_or_array
    if image is None:
        return {"error": "Invalid image"}

    # Preprocess: HSV filter (same as original)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80], dtype=np.uint8)
    upper = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    filtered_img = cv2.bitwise_and(image, image, mask=mask)

    # Classification: Is input correct document type?
    results = model_classifier(filtered_img)
    is_dl_card = any(r.probs.top1 == 0 for r in results)
    if not is_dl_card:
        return {"error": "Please upload appropriate image"}

    # Orientation correction
    result = model_orient(filtered_img)
    predicted_class = result[0].probs.top1
    rotated_image = correct_orientation(filtered_img, predicted_class)

    # Crop document region
    cropped_img = crop_segment(rotated_image, model_card_detector)
    if cropped_img is None:
        return {"error": "No object detected"}

    # Extract text fields
    info_dict = get_details(cropped_img)
    return info_dict









    
