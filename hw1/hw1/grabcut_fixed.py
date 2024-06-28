import numpy as np
import cv2
import argparse
import igraph as ig
import sklearn
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel
neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
beta = 0
gamma, lamda = 15, 5 * 9


# Utility function to visualize data
def visualize_data(data, title):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    if data.ndim == 2:
        plt.imshow(data, cmap='gray')
    else:
        plt.imshow(data)
    plt.colorbar()
    plt.show()


def getNeighborsEdges(img):
    def vid(i, j):  # vertex ID
        return (img.shape[1] * i) + j

    def compute_V(i, j, oi, oj):
        diff = img[i, j] - img[oi, oj]
        return gamma * np.exp(- beta * diff.dot(diff))

    def add_neighbors(row, col, E, W):
        for k, l in neighbors:
            o_i = row + k
            o_j = col + l
            if 0 <= o_i < img.shape[0] and 0 <= o_j < img.shape[1]:
                E.append((vid(i, j), vid(o_i, o_j)))
                W.append(compute_V(i, j, o_i, o_j))

    edges = []
    weights = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # add edges to neighbours
            add_neighbors(i, j, edges, weights)
    return edges, weights


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    start_time = time.perf_counter()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    k = 0

    # Convert from absolute coordinates
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    beta = get_beta(img)
    size_of_fg = img[mask == GC_PR_FGD].reshape((-1, img.shape[-1])).shape[0]
    num_iters = 15
    n_e, n_w = getNeighborsEdges(img)

    for i in range(num_iters):
        #Update GMM
        start_time_iter = time.perf_counter()

        print("iter {}".format(i))

        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM, n_e, n_w)

        # Visualize the likelihoods or energy maps (example)
        #fg_likelihoods = fgGMM.score_samples(img.reshape((-1, 3))).reshape(img.shape[:2])
        #bg_likelihoods = bgGMM.score_samples(img.reshape((-1, 3))).reshape(img.shape[:2])
        #visualize_data(fg_likelihoods, f'Foreground Likelihoods at Iteration {i}')
        #visualize_data(bg_likelihoods, f'Background Likelihoods at Iteration {i}')

        mask = update_mask(mincut_sets, mask)

        temp_size = img[mask == GC_PR_FGD].reshape((-1, img.shape[-1])).shape[0]

        visualize_mask(mask, i)

        if check_convergence(abs(temp_size - size_of_fg)):
            if k > 2:
                break
            k += 1

        size_of_fg = temp_size
        end_time_iter = time.perf_counter()
        elapsed_time_iter = end_time_iter - start_time_iter
        print(f"Elapsed time: {elapsed_time_iter} seconds")

    # Return the final mask and the GMMs
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 60} minutes")

    return mask, bgGMM, fgGMM


# question 2.1 - Amir - Should be OK
def initalize_GMMs(img, mask, n_components=5):
    bg_pixels = img[mask == GC_BGD].reshape((-1, img.shape[-1]))
    fg_pr_pixels = img[mask == GC_PR_FGD].reshape((-1, img.shape[-1]))

    # Adjust the number of components based on available pixels
    actual_components = min(n_components, len(fg_pr_pixels))

    bgGMM = GaussianMixture(n_components)
    fgGMM = GaussianMixture(actual_components)

    bgGMM.fit(bg_pixels)
    if len(fg_pr_pixels) >= actual_components:
        fgGMM.fit(fg_pr_pixels)
    else:
        print("Not enough foreground pixels to initialize the foreground GMM.")
        # Optionally adjust the mask here to increase foreground pixels
        # Example: Expand the foreground area slightly

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
# question 2.2 - Amir - Should be OK
def update_GMMs(img, mask, bgGMM, fgGMM):
    bg_pixels = img[mask == GC_BGD].reshape((-1, img.shape[-1]))
    fg_pr_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)].reshape((-1, img.shape[-1]))

    print(img[mask == GC_BGD].reshape((-1, img.shape[-1])).shape,
          img[mask == GC_PR_BGD].reshape((-1, img.shape[-1])).shape)
    print(img[mask == GC_FGD].reshape((-1, img.shape[-1])).shape,
          img[mask == GC_PR_FGD].reshape((-1, img.shape[-1])).shape)

    bgGMM.fit(bg_pixels)
    if len(fg_pr_pixels) >= 5:
        fgGMM.fit(fg_pr_pixels)
    else:
        print("Not enough foreground pixels to update the foreground GMM.")

    return bgGMM, fgGMM


##def get_beta(img):
##    rows, cols, channels = img.shape
##    beta = 0
##
##    # Calculate squared differences between neighboring pixels for all channels
##    if img.ndim == 3:  # For color images
##        for c in range(channels):
##            left_diff = (img[1:, :, c] - img[:-1, :, c]) ** 2
##            right_diff = (img[:, 1:, c] - img[:, :-1, c]) ** 2
##            down_diff = (img[1:, 1:, c] - img[:-1, :-1, c]) ** 2
##            up_diff = (img[:-1, 1:, c] - img[1:, :-1, c]) ** 2
##
##            beta += np.sum(left_diff) + np.sum(right_diff) + np.sum(down_diff) + np.sum(up_diff)
##    else:  # For grayscale images
##        left_diff = (img[1:, :] - img[:-1, :]) ** 2
##        right_diff = (img[:, 1:] - img[:, :-1]) ** 2
##        down_diff = (img[1:, 1:] - img[:-1, :-1]) ** 2
##        up_diff = (img[:-1, 1:] - img[1:, :-1]) ** 2
##
##        beta = np.sum(left_diff) + np.sum(right_diff) + np.sum(down_diff) + np.sum(up_diff)
##
##    # Normalize by the number of pixel differences considered
##    total_pixels = (rows - 1) * cols + (cols - 1) * rows + (rows - 1) * (cols - 1) * 2
##    beta = 1 / (2 * beta / total_pixels)
##
##    return beta


# Helper function to get beta (smoothness)
def get_beta(img):
    rows, cols, _ = img.shape
    _left_diff = img[:, 1:] - img[:, :-1]
    _upleft_diff = img[1:, 1:] - img[:-1, :-1]
    _up_diff = img[1:, :] - img[:-1, :]
    _upright_diff = img[1:, :-1] - img[:-1, 1:]

    beta = np.sum(np.square(_left_diff)) + np.sum(np.square(_upleft_diff)) + \
           np.sum(np.square(_up_diff)) + \
           np.sum(np.square(_upright_diff))
    beta = 1 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))
    print(beta)
    return beta


# question 2.3 - Amir - May need changes
def calculate_mincut(img, mask, bgGMM, fgGMM, n_edges, n_weights):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]

    fg_D = - fgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_D = - bgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])

    # closure function to calculate boundary energy
    def compute_V(i, j, oi, oj):
        diff = img[i, j] - img[oi, oj]
        return gamma * np.exp(- beta * diff.dot(diff))

    def vid(i, j):  # vertex ID
        return (img.shape[1] * i) + j

    # BUILD GRAPH
    num_pix = img.shape[0] * img.shape[1]
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pix + 2)
    S = num_pix
    T = num_pix + 1
    # the last two vertices are S and T respectively

    edges = []
    weights = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j] == GC_BGD:
                edges.append((vid(i, j), T))
                weights.append(np.inf)
            elif mask[i, j] == GC_FGD:
                edges.append((vid(i, j), S))
                weights.append(np.inf)
            else:
                edges.append((vid(i, j), S))
                weights.append(bg_D[i, j])

                edges.append((vid(i, j), T))
                weights.append(fg_D[i, j])

    edges.extend(n_edges)
    weights.extend(n_weights)

    graph.add_edges(edges, attributes={'weight': weights})

    cut = graph.st_mincut(S, T, capacity='weight')

    return cut, cut.value


# question 2.4 - Aviv
def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    def ind(idx):  # image index
        return ((idx // img.shape[1]), (idx % img.shape[1]))

    bg_vertices = mincut_sets.partition[0]
    fg_vertices = mincut_sets.partition[1]

    num_pix = img.shape[0] * img.shape[1]

    if num_pix in bg_vertices:
        bg_vertices, fg_vertices = fg_vertices, bg_vertices

    new_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for v in fg_vertices:
        if v not in (num_pix, num_pix + 1):
            if mask[ind(v)] == GC_FGD:
                new_mask[ind(v)] = GC_FGD
            else:
                new_mask[ind(v)] = GC_PR_FGD

    return new_mask


# question 2.5 - Amir - Done
def check_convergence(energy):
    print(energy)
    return energy < 25


# question 2.6 - Amir - Done
def cal_metric(predicted_mask, gt_mask):
    intersection = np.logical_and(predicted_mask, gt_mask)
    union = np.logical_or(predicted_mask, gt_mask)
    jaccard = np.sum(intersection) / np.sum(union)
    accuracy = np.sum(predicted_mask == gt_mask) / gt_mask.size
    return accuracy, jaccard


def visualize_mask(mask, iteration):
    plt.figure(figsize=(10, 5))
    plt.title(f'Mask State at Iteration {iteration}')
    plt.imshow(mask, cmap='gray')
    plt.colorbar()
    plt.show()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='stone2', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
