import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse

def poisson_blend(im_src, im_tgt, im_mask, center):
    x, y = center
    h, w = im_mask.shape

    # Compute the region of interest in the target image
    y1, y2 = y - h // 2, y + h // 2
    x1, x2 = x - w // 2, x + w // 2

    if y1 < 0 or y2 > im_tgt.shape[0] or x1 < 0 or x2 > im_tgt.shape[1]:
        raise ValueError("The source image is too large to fit in the target image at the specified offset")

    im_tgt_crop = im_tgt[y1:y2, x1:x2]

    if im_tgt_crop.shape[:2] != im_src.shape[:2]:
        raise ValueError("The cropped target image must have the same dimensions as the source image")

    # Create the Laplacian operator
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])

    # Compute the gradient of the source image
    gradient = cv2.filter2D(im_src, cv2.CV_64F, laplacian)

    # Initialize the blended region with the target region
    blended = im_tgt_crop.astype(np.float64)

    # Iteratively solve the Poisson equation using the Jacobi method
    for _ in range(2000):
        blended[1:-1, 1:-1] = (blended[:-2, 1:-1] + blended[2:, 1:-1] +
                               blended[1:-1, :-2] + blended[1:-1, 2:] -
                               gradient[1:-1, 1:-1]) / 4.0
        blended[im_mask == 0] = im_tgt_crop[im_mask == 0]

    # Convert the blended region back to the original image format
    blended = blended.clip(0, 255).astype(im_tgt.dtype)

    # Replace the region in the target image with the blended region
    result = im_tgt.copy()
    result[y1:y2, x1:x2] = blended

    return result

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape[:2], 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    # Ensure that the mask and source image have the same dimensions
    if im_mask.shape[:2] != im_src.shape[:2]:
        raise ValueError("The mask and source image must have the same dimensions")

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.imwrite('blended_output_refined.png', im_clone)
    print('Blended image saved as blended_output_refined.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
