import sys
import cv2

img_path = '/media/wangnan/Data/JPEGImages/000048.jpg'
gt_path = '/home/wangnan/Append/Project/Underwater_UNet/test/gen_fog4/000048_gen.png'

img_gen = cv2.imread(img_path)
img_gt = cv2.imread(gt_path)
cv2.namedWindow("ground truth")
cv2.namedWindow("before resize")
cv2.namedWindow("after resize")

if img_gen is None:
    print("could not read image!")
    sys.exit(0)

img_size = (256, 256)
print(img_size)

new_image = cv2.resize(img_gen, img_size, interpolation=cv2.INTER_AREA)

cv2.imshow("ground truth", img_gt)
cv2.imshow("before resize", img_gen)
cv2.imshow("after resize", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()




