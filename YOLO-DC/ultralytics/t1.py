from ultralytics import YOLO
from PIL import Image
import cv2

# model = YOLO("./runs/detect/train18/weights/best.pt")
model = YOLO("./runs/detect/train18/weights/best.pt")
results = model.predict('../fish_test/videos/VCG42N1049754888.mp4', save=True)
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
#
# # from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images
#
# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])