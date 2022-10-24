import cv2
import rawpy
import numpy as np
import glob
import torch
from tqdm import tqdm


def single_calibrate(images, width=11, height=8, square_size = 30, shrink_factor = 0.5):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if square_size is None:
        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    else:
        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width*square_size:square_size, 0:height*square_size:square_size].T.reshape(-1, 2)
        
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane
    
    idx = 0
    for img in tqdm(images):
        img = cv2.resize(img, (0,0), fx=shrink_factor, fy=shrink_factor)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function.
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

            # cv2.imwrite(path+str(idx)+'.jpg', img)
            idx += 1

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs



def stereo_calibrate(images1, images2, mtx1=None, mtx2=None, dist1=None, dist2=None, width=11, height=8, square_size = 30, shrink_factor = 0.5):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if square_size is None:
        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    else:
        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width*square_size:square_size, 0:height*square_size:square_size].T.reshape(-1, 2)

    objpoints1 = [] # 3d point in real world space
    objpoints2 = [] 
    imgpoints1 = [] # 2d points in image plane
    imgpoints2 = []

    objpoints1_single = [] # 3d point in real world space
    objpoints2_single = [] 
    imgpoints1_single = [] # 2d points in image plane
    imgpoints2_single = []

    idx = 0
    for image1, image2 in tqdm(zip(images1, images2)):
        image1 = cv2.resize(image1, (0,0), fx=shrink_factor, fy=shrink_factor)
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        image2 = cv2.resize(image2, (0,0), fx=shrink_factor, fy=shrink_factor)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret1, corners1 = cv2.findChessboardCorners(gray1, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (width, height), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret1 and ret2:
            objpoints1.append(objp)
            objpoints1_single.append(objp)
            objpoints2.append(objp)
            objpoints2_single.append(objp)

            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            imgpoints1.append(corners1)
            imgpoints1_single.append(corners1)
            imgpoints2.append(corners2)
            imgpoints2_single.append(corners2)

            image1_cb = cv2.drawChessboardCorners(image1, (width, height), corners1, ret1)
            image2_cb = cv2.drawChessboardCorners(image2, (width, height), corners2, ret2)
            cv2.imwrite(path +'cb/left'+str(idx)+'.jpg', image1_cb)
            cv2.imwrite(path +'cb/right'+str(idx)+'.jpg', image2_cb)

        elif ret1:
            objpoints1_single.append(objp)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            imgpoints1_single.append(corners1)
            
            image1_cb = cv2.drawChessboardCorners(image1, (width, height), corners1, ret1)
            cv2.imwrite(path +'cb/left'+str(idx)+'.jpg', image1_cb)
        elif ret2:
            objpoints2_single.append(objp)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            imgpoints2_single.append(corners2)
            
            image2_cb = cv2.drawChessboardCorners(image2, (width, height), corners2, ret2)
            cv2.imwrite(path +'cb/right'+str(idx)+'.jpg', image2_cb)
            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function.
            # img1 = cv2.drawChessboardCorners(image1, (width, height), corners1, ret1)
            # img2 = cv2.drawChessboardCorners(image2, (width, height), corners2, ret2)
            # cv2.imshow('img1', img1)
            # cv2.imshow('img2', img2)
            # cv2.waitKey(0)

        idx += 1 
    
    if all([mtx1, mtx2, dist1, dist2]):
        pass
    elif all([mtx1, dist1]):
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2_single, imgpoints2_single, gray2.shape[::-1], None, None)
    elif all([mtx2, dist2]):
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1_single, imgpoints1_single, gray1.shape[::-1], None, None)
    else:
        # single camera calibration
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1_single, imgpoints1_single, gray1.shape[::-1], None, None)
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2_single, imgpoints2_single, gray2.shape[::-1], None, None)

    if ret1 and ret2:
        # stereo calibration
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints1, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5)

    return retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F


def correct_distortion(image, mtx, dist):
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    return dst

# calculate world coordinates of the image points based on 2 camera calibration results
def recover_world_coords(left_points, right_points, left_mtx, right_mtx, R, T):
    if all([left_points, right_points, left_mtx, right_mtx, R, T]):
        mat_rt = np.hstack((R, T))
        tsr_cam_l = torch.FloatTensor(np.concatenate((left_mtx, [[0], [0], [0]]), axis=1))
        tsr_cam_r = torch.FloatTensor(np.concatenate((right_mtx, [[0], [0], [0]]), axis=1)@mat_rt)
        tsr_pt_l = torch.FloatTensor(left_points)
        tsr_pt_r = torch.FloatTensor(right_points)
        tsr_fml = torch.stack((
            tsr_pt_l[:, 0:1]*tsr_cam_l[2:3]-tsr_cam_l[0:1],
            tsr_pt_l[:, 1:2]*tsr_cam_l[2:3]-tsr_cam_l[1:2],
            tsr_pt_r[:, 0:1]*tsr_cam_r[2:3]-tsr_cam_r[0:1],
            tsr_pt_r[:, 1:2]*tsr_cam_r[2:3]-tsr_cam_r[1:2],
        ))
        tsr_fml = tsr_fml.transpose(1, 0)
        tsr_root = torch.svd(tsr_fml)[2][:, :, -1]
        world_points = (tsr_root[:, :3]/tsr_root[:, 3:4]).numpy().astype(np.float64)

    return world_points

def extract_aruco(gray, SUBPIX = False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    params = cv2.aruco.DetectorParameters_create()
    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    markers, ids, _ = cv2.aruco.detectMarkers(gray, dictionary=dict,parameters=params)

    if SUBPIX:
        markers = list(markers)
        w = int(abs(markers[0][0][1][0] - markers[0][0][0][0]) // 20)
        if w % 2 == 0:
            w -= 1
        if w >= 3:
            for i in range(len(markers)):
                markers[i] = cv2.cornerSubPix(gray, markers[i], (w, w), (-1, -1), criteria)
    return tuple(markers), ids


if __name__ == "__main__":

    # MODE = 'single'
    MODE = 'stereo'

    path = r"D:\images\20220616chessboard"
    square_size = 20
    shrink_factor = 0.5
    width = 11
    height = 8

    if MODE == 'single':
        images = []
        print('loading images...')
        for f in tqdm(sorted(glob.glob(path + "*.RAF"))):
            raw = rawpy.imread(f)
            img = raw.postprocess(gamma = (1, 1), no_auto_bright = True, use_camera_wb = True)
            images.append(img)

        ret, mtx, dist, rvecs, tvecs, _ = single_calibrate(images, width=width, height=height, square_size=square_size, shrink_factor=shrink_factor)
        
        print('ret:', ret)
        print('mtx: ', mtx)
        print('dist: ', dist)
        print('rvecs: ', rvecs)
        print('tvexs: ', tvecs)

        raw = rawpy.imread(path + 'DSCF0043.RAF')
        image = raw.postprocess(gamma = (1, 1), no_auto_bright = True, output_bps = 16, use_camera_wb = True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_undist = correct_distortion(image, mtx, dist)
        cv2.imwrite(path+'image_undist.png', image_undist)

    elif MODE == 'stereo':
        images1 = []
        images2 = []
        print('loading images...')
        for file in tqdm(sorted(glob.glob(path + '/mid/*.RAF'))):
            raw = rawpy.imread(file)
            img = raw.postprocess(gamma = (1, 1), no_auto_bright = True, use_camera_wb = True)
            images1.append(img)

        for file in tqdm(sorted(glob.glob(path + '/right/*.RAF'))):
            raw = rawpy.imread(file)
            img = raw.postprocess(gamma = (1, 1), no_auto_bright = True, use_camera_wb = True)
            images2.append(img)

        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = stereo_calibrate(images1, images2, width=width, height=height, square_size = square_size, shrink_factor = shrink_factor)

        print('ret:', retval)
        print('mtx1: ', cameraMatrix1)
        print('mtx2: ', cameraMatrix2)
        print('dist1: ', distCoeffs1)
        print('dist2: ', distCoeffs2)
        print('rvecs: ', R)
        print('tvexs: ', T)

    exit(0)