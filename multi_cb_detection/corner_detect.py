import cv2
import numpy as np
import math

class cornerDetector:
    def __init__(self, img):
        radius = np.array([4, 8, 12])
        template_props = [[0, math.pi/2], [math.pi/4, -math.pi/4]]
        template_props = np.array(template_props * 3)

    def __norm_pdf(self, dist, mu, sigma):
        s = math.exp(-0.5*((dist-mu)/sigma)**2)
        return s/(sigma*math.sqrt(2*math.pi))

    def __create_kernel(self, angle1, angle2, kernel_size):
        width = kernel_size * 2 + 1
        height = kernel_size * 2 + 1
        kernelA = np.zeros((width, height))
        kernelB = np.zeros((width, height))
        kernelC = np.zeros((width, height))
        kernelD = np.zeros((width, height))

        for u in range(width):
            for v in range(height):
                vec = np.array([u - kernel_size, v - kernel_size])
                dis = np.linalg.norm(vec)
                side1 = np.dot(vec, np.array([-math.sin(angle1), math.cos(angle1)]))
                side2 = np.dot(vec, np.array([-math.sin(angle2), math.cos(angle2)]))
                if side1 <= -0.1 and side2 <= -0.1:
                    kernelA[u, v] = self.__norm_pdf(dis, 0, kernel_size/2)
                elif side1 >= 0.1 and side2 >= 0.1:
                    kernelB[u, v] = self.__norm_pdf(dis, 0, kernel_size/2)
                elif side1 <= -0.1 and side2 >= 0.1:
                    kernelC[u, v] = self.__norm_pdf(dis, 0, kernel_size/2)
                elif side1 >= 0.1 and side2 <= -0.1:
                    kernelD[u, v] = self.__norm_pdf(dis, 0, kernel_size/2)

        kernelA = kernelA / np.sum(kernelA)
        kernelB = kernelB / np.sum(kernelB)
        kernelC = kernelC / np.sum(kernelC)
        kernelD = kernelD / np.sum(kernelD)
        return kernelA, kernelB, kernelC, kernelD

    def __get_min(src1, src2):
        return np.minimum(src1, src2)
    
    def __get_max(src1, src2):
        return np.maximum(src1, src2)
    
    def __get_image_angle_and_weight(self, img):
        img_du = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        img_dv = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        if img_du.shape != img_dv.shape:
            raise ValueError("img_du and img_dv must have the same shape")
        
        img_weight, img_angle = cv2.cartToPolar(img_du, img_dv, angleInDegrees=False)

        img_angle[img_angle < 0] += math.pi
        img_angle[img_angle > math.pi] -= math.pi

        return img_du, img_dv, img_angle, img_weight
    
    def __non_max_suppression(self, input_corners, patch_size, threshold, margin):
        output_corners = []
        for i in range(margin+patch_size, input_corners.shape[1]-margin-patch_size, patch_size+1):
            for j in range(margin+patch_size, input_corners.shape[0]-margin-patch_size, patch_size+1):
                # find the max value and location of patch
                max_y, max_x = np.argmax(input_corners[j:j+patch_size, i:i+patch_size])
                max_val = input_corners[max_y, max_x]
                if max_val < threshold:
                    continue
                
