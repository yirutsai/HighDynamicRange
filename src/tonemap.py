import cv2
import os
import numpy as np

p = ("d:/scott/SCOTT/NTUME/VFX/HW1/result/img1/produce.hdr")
hdr = cv2.imread(p,cv2.IMREAD_UNCHANGED)
print(hdr.shape)
# hdr = hdr[:,:,0]
print(hdr.shape)
print(hdr.dtype)
hdr = hdr.astype(np.float32)
print(hdr.dtype)

output_path = "d:/scott/SCOTT/NTUME/VFX/HW1/result/img1/ldr3.png"

tonemap = cv2.createTonemap(2.2)
ldr = tonemap.process(hdr)

cv2.imwrite(output_path, ldr *255)