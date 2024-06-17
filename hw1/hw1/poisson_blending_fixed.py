import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse

def poisson_blend(im_src, im_tgt, im_mask, center):
    h_src, w_src = im_src.shape[:2]
    h_tgt, w_tgt = im_tgt.shape[:2]

    # Calculate offsets correctly
    y_offset = center[1] - h_src // 2
    x_offset = center[0] - w_src // 2

    y_offset = max(0, min(y_offset, h_tgt - h_src))
    x_offset = max(0, min(x_offset, w_tgt - w_src))

    # print(f"y_offset: {y_offset}, x_offset: {x_offset}")
    # print(f"ROI size in target image: ({y_offset}, {y_offset + h_src}), ({x_offset}, {x_offset + w_src})")

    # Extract the region of interest from the target image
    tgt_roi = im_tgt[y_offset:y_offset + h_src, x_offset:x_offset + w_src].copy()

    # Construct Laplacian matrix and b vector
    mask_indices = np.where(im_mask > 0)
    num_pixels = mask_indices[0].shape[0]

    # print(f"Number of masked pixels: {num_pixels}")

    A = scipy.sparse.lil_matrix((num_pixels, num_pixels))
    b = np.zeros((num_pixels, im_src.shape[2]))

    for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
        A[i, i] = 4
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            yy, xx = y + dy, x + dx
            if 0 <= yy < im_mask.shape[0] and 0 <= xx < im_mask.shape[1]:
                if im_mask[yy, xx] > 0:
                    j = np.where((mask_indices[0] == yy) & (mask_indices[1] == xx))[0][0]
                    A[i, j] = -1
                else:
                    b[i] += tgt_roi[yy - y_offset, xx - x_offset]

    blended = tgt_roi.copy()
    for channel in range(im_src.shape[2]):
        f = im_src[mask_indices[0], mask_indices[1], channel]
        # print(f"Solving Poisson equation for channel {channel}")
        try:
            f_prime = spsolve(A.tocsc(), f - b[:, channel])
        except RuntimeError as e:
            # print(f"Runtime error during spsolve for channel {channel}: {e}")
            continue
        blended[mask_indices[0], mask_indices[1], channel] = np.clip(f_prime, 0, 255)

    im_tgt[y_offset:y_offset + h_src, x_offset:x_offset + w_src] = blended

    return im_tgt

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='C:/Users/Aviv/Downloads/Tau/graphics/hw1/hw1/data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='C:/Users/Aviv/Downloads/Tau/graphics/hw1/hw1/data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='C:/Users/Aviv/Downloads/Tau/graphics/hw1/hw1/data/bg/table.jpg', help='target image file path')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape[:2], 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    # print(f"Source image shape: {im_src.shape}")
    # print(f"Target image shape: {im_tgt.shape}")
    # print(f"Mask shape: {im_mask.shape}")

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imwrite('blended_output_refined.png', im_clone)
    # print('Blended image saved as blended_output_refined.png')
