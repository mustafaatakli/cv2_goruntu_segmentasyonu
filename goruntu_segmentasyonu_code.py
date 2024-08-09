import cv2
import numpy as np
class GrabCut:
    def __init__(self):
        image_path = 'image.JPG'
        self.image = cv2.imread(image_path)
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self.background_model = np.zeros((1, 65), np.float64)
        self.front_model = np.zeros((1, 65), np.float64)

    def masking(self):
        self.mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.result = self.image * self.mask2[:, :, np.newaxis]

    def rectangle(self, rect):
        self.rect = rect

    def segmentation(self):
        cv2.grabCut(self.image, self.mask, self.rect, self.background_model, self.front_model, 7, cv2.GC_INIT_WITH_RECT)
        self.masking()

    def result_image(self):
        cv2.imshow('Result-image', self.result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    grabcut = GrabCut()
    rect = (60, 40, 650, 650)
    grabcut.rectangle(rect)
    grabcut.segmentation()
    grabcut.result_image()

main()