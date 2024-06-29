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
gamma, lamda = 5, 5 * 9


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
    rows, cols = img.shape[:2]
    vid = lambda i, j: i * cols + j

    # Define neighbor offsets for all 8 directions
    neighbor_offsets = np.array([
        [0, 1],  # Right
        [1, 0],  # Down
        [0, -1],  # Left
        [-1, 0],  # Up
        [1, 1],  # Down-right
        [1, -1],  # Down-left
        [-1, -1],  # Up-left
        [-1, 1]  # Up-right
    ])

    # Compute vertex IDs
    indices = np.indices((rows, cols)).reshape(2, -1)

    # Prepare empty lists for edges and weights
    edges = []
    weights = []

    # Pre-compute squared differences for efficiency
    def compute_V(i, j, oi, oj):
        diff = img[i, j] - img[oi, oj]
        return gamma * np.exp(-beta * np.dot(diff, diff))

    for offset in neighbor_offsets:
        neighbor_indices = indices + offset[:, None]
        valid_mask = (
            (neighbor_indices[0] >= 0) & (neighbor_indices[0] < rows) &
            (neighbor_indices[1] >= 0) & (neighbor_indices[1] < cols)
        )
        valid_neighbors = neighbor_indices[:, valid_mask]
        valid_vertices = indices[:, valid_mask]

        # Compute edges
        from_vids = vid(valid_vertices[0], valid_vertices[1])
        to_vids = vid(valid_neighbors[0], valid_neighbors[1])
        edges.extend(zip(from_vids, to_vids))

        # Compute weights
        for v, nv in zip(valid_vertices.T, valid_neighbors.T):
            weights.append(compute_V(v[0], v[1], nv[0], nv[1]))

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
    visualize_mask(mask, "start")

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    beta = get_beta(img)
    size_of_fg = img[mask == GC_PR_FGD].reshape((-1, img.shape[-1])).shape[0]
    num_iters = 30
    n_e, n_w = getNeighborsEdges(img)

    for i in range(num_iters):
        #Update GMM
        start_time_iter = time.perf_counter()

        print("iter {}".format(i))
        start_time_func1 = time.perf_counter()
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        end_time_func = time.perf_counter()
        elapsed_time_func1 = end_time_func - start_time_func1
        #print(f"Elapsed time of update GMM: {elapsed_time_func1} seconds")

        start_time_func1 = time.perf_counter()
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM, n_e, n_w)
        end_time_func = time.perf_counter()
        elapsed_time_func1 = end_time_func - start_time_func1
        #print(f"Elapsed time of mincut : {elapsed_time_func1} seconds")

        mask = update_mask(mincut_sets, mask)

        temp_size = img[mask == GC_PR_FGD].reshape((-1, img.shape[-1])).shape[0]

        visualize_mask(mask, i)

        if check_convergence(abs(temp_size - size_of_fg)):
            k += 1
            if k > 1:
                break

        size_of_fg = temp_size
        end_time_iter = time.perf_counter()
        elapsed_time_iter = end_time_iter - start_time_iter
        print(f"Elapsed time: {elapsed_time_iter} seconds")

        if end_time_iter - start_time > 175:
            break

    # Return the final mask and the GMMs
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 60} minutes")

    return mask, bgGMM, fgGMM


# question 2.1 - Amir - Should be OK
def initalize_GMMs(img, mask, n_components=2):
    bg_pixels = img[mask == GC_BGD].reshape((-1, img.shape[-1]))
    fg_pr_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)].reshape((-1, img.shape[-1]))

    # Adjust the number of components based on available pixels
    actual_components = min(n_components, len(fg_pr_pixels))

    bgGMM = GaussianMixture(n_components, covariance_type='full')
    fgGMM = GaussianMixture(actual_components, covariance_type='full')

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
    bg_pixels = img[mask == GC_BGD]
    fg_pr_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)]

    bg_pixels = bg_pixels.reshape(-1, img.shape[-1])
    fg_pr_pixels = fg_pr_pixels.reshape(-1, img.shape[-1])

    bgGMM.fit(bg_pixels)

    if fg_pr_pixels.shape[0] >= 4:
        fgGMM.fit(fg_pr_pixels)
    else:
        print("Not enough foreground pixels to update the foreground GMM.")

    return bgGMM, fgGMM


# Helper function to get beta (smoothness)
def get_beta(img):
    rows, cols, _ = img.shape
    diff_squares = (np.square(img[:, 1:] - img[:, :-1]).sum() +
                    np.square(img[1:, 1:] - img[:-1, :-1]).sum() +
                    np.square(img[1:, :] - img[:-1, :]).sum() +
                    np.square(img[1:, :-1] - img[:-1, 1:]).sum())
    beta = 1 / (2 * diff_squares / (4 * cols * rows - 3 * cols - 3 * rows + 2))
    print("beta = ", beta)
    return beta


# question 2.3 - Amir - May need changes
def calculate_mincut(img, mask, bgGMM, fgGMM, n_edges, n_weights):
    # TODO: implement energy (cost) calculation step and mincut

    fg_D = - fgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_D = - bgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])

    # closure function to calculate boundary energy
    def compute_V(i, j, oi, oj):
        diff = img[i, j] - img[oi, oj]
        return gamma * np.exp(- beta * diff.dot(diff))

    def vid(i, j):  # vertex ID
        return (img.shape[1] * i) + j

    # graph
    num_pix = img.shape[0] * img.shape[1]
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pix + 2)
    S = num_pix
    T = num_pix + 1
    # the last two vertices are S and T respectively

    edges = []
    weights = []

    # Vectorized approach to get all indices
    indices = np.indices((img.shape[0], img.shape[1]))
    i_indices = indices[0].flatten()
    j_indices = indices[1].flatten()

    # Flattened mask
    flat_mask = mask.flatten()

    # Process background and foreground
    bg_indices = np.where(flat_mask == GC_BGD)
    fg_indices = np.where(flat_mask == GC_FGD)
    other_indices = np.where((flat_mask != GC_BGD) & (flat_mask != GC_FGD))

    # Append edges and weights for background
    bg_vid = [vid(i, j) for i, j in zip(i_indices[bg_indices], j_indices[bg_indices])]
    edges.extend([(v, T) for v in bg_vid])
    weights.extend([np.inf] * len(bg_vid))

    # Append edges and weights for foreground
    fg_vid = [vid(i, j) for i, j in zip(i_indices[fg_indices], j_indices[fg_indices])]
    edges.extend([(v, S) for v in fg_vid])
    weights.extend([np.inf] * len(fg_vid))

    # Append edges and weights for other cases
    other_vid = [vid(i, j) for i, j in zip(i_indices[other_indices], j_indices[other_indices])]
    other_bg_D = bg_D.flatten()[other_indices]
    other_fg_D = fg_D.flatten()[other_indices]

    edges.extend([(v, S) for v in other_vid])
    weights.extend(other_bg_D.tolist())

    edges.extend([(v, T) for v in other_vid])
    weights.extend(other_fg_D.tolist())

    # Extend with n_edges and n_weights
    edges.extend(n_edges)
    weights.extend(n_weights)

    # Add edges to the graph
    graph.add_edges(edges, attributes={'weight': weights})

    # Compute the mincut
    cut = graph.st_mincut(S, T, capacity='weight')

    return cut, cut.value


# question 2.4 - Aviv
def update_mask(mincut_sets, mask):
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
    print(f"Total energy: {energy}")
    return energy < 10


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
    parser.add_argument('--input_name', type=str, default='book', help='name of image from the course files')
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
