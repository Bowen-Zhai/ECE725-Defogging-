import cv2
import numpy as np
import sys
import kdtree
import rgbProcessing
import wlsFilter
import regularization
import config
from retinex import MSRCR
filter_size = 15
p = 0


def non_local_transmission(img, air, gamma=1):
    img_hazy_corrected = np.power(img, gamma)
    dist_from_airlight = rgbProcessing.getDistAirlight(img_hazy_corrected, air)
    row, col, n_colors = img.shape
    radius = np.sqrt(np.sum(dist_from_airlight ** 2, axis=2))
    dist_sphere_radius = np.reshape(radius, [col * row], order='F')
    dist_unit_radius = np.reshape(dist_from_airlight, [col * row, n_colors], order='F')
    dist_norm = np.sqrt(np.sum(dist_unit_radius ** 2, axis=1))
    for i in range(len(dist_unit_radius)):
        dist_unit_radius[i] = dist_unit_radius[i] / dist_norm[i]

    n_points = 1000
    file_path = "./TR" + str(n_points) + ".txt"
    points = np.loadtxt(file_path).tolist()
    mdl = kdtree.create(points)
    cluster = [[]] * n_points
    for i in range(n_points):
        cluster[i] = []

    cluster_Points = np.zeros(row * col, dtype=int)
    for r in range(len(dist_unit_radius)):
        kdNode = mdl.search_knn(dist_unit_radius[r], 1)
        findPosition(kdNode[0][0].data, dist_sphere_radius[r], cluster, points, r, cluster_Points)
    maxRadius = np.zeros(row * col, dtype=float)
    for i in range(n_points):
        maxR = 0
        for j in range(len(cluster[i])):
            maxR = max(maxR, cluster[i][j])
        maxRadius[i] = maxR
    np.reshape(maxRadius, [row, col], order='F')
    dist_sphere_maxRadius = np.zeros(row * col, float)
    for i in range(row * col):
        index = cluster_Points[i]
        dist_sphere_maxRadius[i] = maxRadius[index]
    radius_new = np.reshape(dist_sphere_maxRadius, [row, col], order='F')
    transmission_estimation = radius / (radius_new + p)
    trans_min = 0.1
    transmission_estimation = np.minimum(np.maximum(transmission_estimation, trans_min), 1)
    transmission = regularization.regularization(row, col, transmission_estimation, img_hazy_corrected, n_points, air,
                                                 cluster_Points, cluster)
    return transmission

def findPosition(kdNode, radius, cluster, points, r, cluster_Points):
    for i in range(len(points)):
        if (points[i][0] == kdNode[0]) and (points[i][1] == kdNode[1]) and (points[i][2] == kdNode[2]):
            cluster[i].append(radius)
            cluster_Points[r] = i
            break

def dehaze(img, img_norm, transmission_estimission, air):
    h, w, n_colors = img.shape
    img_dehazed = np.zeros((h, w, n_colors), dtype=float)
    leave_haze = 1.06
    for color_idx in range(3):
        img_dehazed[:, :, color_idx] = (img_norm[:, :, color_idx] - (1 - leave_haze * transmission_estimission) * air[
            color_idx]) / np.maximum(transmission_estimission, 0.1)

    img_dehazed = np.where(img_dehazed > 1, 1, img_dehazed)
    img_dehazed = np.where(img_dehazed < 0, 0, img_dehazed)
    img_dehazed = np.power(img_dehazed, 1 / 1)
    adj_percent = [0.005, 0.995]
    # img_dehazed = adjust(img_dehazed, adj_percent)
    # img_dehazed = (img_dehazed * 255).astype(np.uint8)
    # print(img_dehazed)
    return img_dehazed

def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

    # Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def main():
    im = []
    path = 'img/'
    for i in range(1,5):
        image = cv2.imread(path + "test" + str(i) + ".jpg")
        im.append(image)

    img = im[2]
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 200)
    height = int(img.shape[0] * scale_percent / 200)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("input_image", img)
    img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
                             cv2.NORM_MINMAX)
    dark = rgbProcessing.dark_channel(img, filter_size)
    air = rgbProcessing.air_light(img, dark)
    air = air[0] / 255
    transmission_estimission = non_local_transmission(img_norm, air)
    clear_img = dehaze(img, img_norm, transmission_estimission, air)
    # cv2.imwrite('clear_img.jpg', clear_img)
    msrcr_img = MSRCR(clear_img, config.SIGMA_LIST, config.ALPHA, config.BETA, config.G, config.OFFSET)
    # img = cv2.imread('clear_img.jpg')
    # auto_result, alpha, beta = automatic_brightness_and_contrast(img)
    cv2.imshow("cleared image", clear_img)
    cv2.imshow("emage enhanced", msrcr_img)
    cv2.imshow("non-local transmission", transmission_estimission)

    # for i in range(len(im)):
    #     img = im[i]
    #     cv2.imshow("input_image", img)
    #     img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
    #                              cv2.NORM_MINMAX)
    #     dark = rgbProcessing.dark_channel(img, filter_size)
    #     air = rgbProcessing.air_light(img, dark)
    #     air = air[0] / 255
    #     transmission_estimission = non_local_transmission(img_norm, air)
    #     clear_img = dehaze(img, img_norm, transmission_estimission, air)
    #     # cv2.imwrite('clear_img.jpg', clear_img)
    #     msrcr_img = MSRCR(clear_img, config.SIGMA_LIST, config.ALPHA, config.BETA, config.G, config.OFFSET)
    #     # img = cv2.imread('clear_img.jpg')
    #     # auto_result, alpha, beta = automatic_brightness_and_contrast(img)
    #     cv2.imshow("cleared image", clear_img)
    #     cv2.imshow("emage enhanced", msrcr_img)
    #     cv2.imshow("non-local transmission", transmission_estimission)

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf)
    sys.setrecursionlimit(100000)
    main()
    cv2.waitKey(0)