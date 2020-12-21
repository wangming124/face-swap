import numpy as np
import cv2


def blendImages(src, dst, mask, featherAmount=0.0):
    # radius = 5  # kernel size
    # kernel = np.ones((radius, radius), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=3)
    maskIndices = np.where(mask != 0)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))

    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
            1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    return composedImg


def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)

    maskIndices = np.where(mask != 0)

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


def blendImages0(src, dst, mask, mask1, featherAmount=0.01):
    # radius = 5  # kernel size
    # kernel = np.ones((radius, radius), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=3)
    maskIndices = np.where(mask != 0)
    maskIndices1 = np.where(mask1 != 0)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    maskPts1 = np.hstack((maskIndices1[1][:, np.newaxis], maskIndices1[0][:, np.newaxis]))
    faceSize0 = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    faceSize = np.max(maskPts1, axis=0) - np.min(maskPts1, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])

    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
            1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    return composedImg


