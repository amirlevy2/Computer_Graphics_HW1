import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def poisson_blend(im_src, im_tgt, im_mask, center):
    # Dimensions of the source and target images
    h_src, w_src = im_src.shape[:2]
    h_tgt, w_tgt = im_tgt.shape[:2]

    # Offsets for placing the source image at the center in the target image
    x_offset = center[1] - h_src // 2
    y_offset = center[0] - w_src // 2

    # ROI in the target image where the source image will be blended
    tgt_roi = im_tgt[x_offset:x_offset + h_src, y_offset:y_offset + w_src]

    # Prepare the Laplacian operator
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

    # Calculate gradients of the source image
    gradient_x = cv2.filter2D(im_src, -1, np.array([[1, -1]]))
    gradient_y = cv2.filter2D(im_src, -1, np.array([[1], [-1]]))

    # Calculate the divergence of gradients
    div_grad = cv2.filter2D(gradient_x, -1, np.array([[1], [-1]])) + \
               cv2.filter2D(gradient_y, -1, np.array([[1, -1]]))

    # Create the sparse matrix for the Laplacian operator
    num_pixels = h_src * w_src
    laplacian = scipy.sparse.lil_matrix((num_pixels, num_pixels))

    # Fill the Laplacian matrix
    for y in range(h_src):
        for x in range(w_src):
            idx = x + y * w_src
            if im_mask[y, x] > 0:
                laplacian[idx, idx] = 4
                if x > 0:
                    laplacian[idx, idx - 1] = -1
                if x < w_src - 1:
                    laplacian[idx, idx + 1] = -1
                if y > 0:
                    laplacian[idx, idx - w_src] = -1
                if y < h_src - 1:
                    laplacian[idx, idx + w_src] = -1

    # Convert the Laplacian matrix to CSC format
    laplacian = laplacian.tocsc()

    # Solve the Poisson equation for each color channel
    blended_channels = []
    for channel in range(3):
        tgt_channel = tgt_roi[:, :, channel].flatten()
        src_channel = im_src[:, :, channel].flatten()
        div_channel = div_grad[:, :, channel].flatten()

        # Setup the system of linear equations
        b = div_channel
        for y in range(h_src):
            for x in range(w_src):
                if im_mask[y, x] == 0:
                    idx = x + y * w_src
                    b[idx] = tgt_channel[idx]

        x = spsolve(laplacian, b)

        # Reshape the solution to the source image shape
        blended_channel = np.clip(x, 0, 255).reshape(h_src, w_src).astype(np.uint8)
        blended_channels.append(blended_channel)

    # Merge the color channels back into an image
    im_blend = cv2.merge(blended_channels)

    # Place the blended region back into the target image
    im_tgt[x_offset:x_offset + h_src, y_offset:y_offset + w_src] = im_blend

    return im_tgt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
