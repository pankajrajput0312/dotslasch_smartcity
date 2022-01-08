import cv2
import numpy as np
# img = cv2.imread("1.png")
# print(img.shape)
img = np.zeros((800,800), dtype = np.float32)
#img2 = cv2.imread("flower.jpg")

# variables
ix = -1
iy = -1
drawing = False

def draw_reactangle_with_drag(event, x, y, flags, param):
    global ix, iy, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, pt1=(ix,iy), pt2=(x, y),color=(255,255,255),thickness=-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, pt1=(ix,iy), pt2=(x, y),color=(255,255,255),thickness=-1)

cv2.namedWindow(winname= "Title of Popup Window")
cv2.setMouseCallback("Title of Popup Window", draw_reactangle_with_drag)

while True:
    cv2.imshow("Title of Popup Window", img)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()