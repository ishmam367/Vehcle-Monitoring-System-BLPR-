from PIL import Image
import pytesseract
import cv2


img= cv2.imread('plate1.png')
tex=pytesseract.image_to_string(Image.open('plate1.png'),lang='ben')
print(pytesseract.image_to_string(Image.open('plate1.png'),lang='ben'))
cv2.nameWindow('Input image')
cv2.imshow("Input image",img)
cv2.waitkey(0)
cv2.destroyWindow("Test")
cv2.destroyWindow("Main")
