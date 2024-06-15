import numpy as np
import cv2
import argparse
import igraph as ig
import sklearn
from sklearn.mixture import GaussianMixture

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel
neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
beta = 0
gamma, lamda = 50, 50 * 9


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute coordinates
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    beta = get_beta(img)

    num_iters = 15
    for i in range(num_iters):
        #Update GMM
        print("iter {}".format(i))
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


# question 2.1 - Amir - Should be OK
def initalize_GMMs(img, mask):
    # TODO: implement initialize_GMMs
    bg_pixels = img[mask == GC_BGD].reshape((-1, img.shape[-1]))
    fg_pr_pixels = img[mask == GC_PR_FGD].reshape((-1, img.shape[-1]))

    bgGMM = GaussianMixture(n_components=5)
    fgGMM = GaussianMixture(n_components=5)

    bgGMM.fit(bg_pixels)
    fgGMM.fit(fg_pr_pixels)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
# question 2.2 - Amir - Should be OK
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    bg_pixels = img[mask == GC_BGD].reshape((-1, img.shape[-1]))
    fg_pr_pixels = img[mask == GC_PR_FGD].reshape((-1, img.shape[-1]))

    print(img[mask == GC_BGD].reshape((-1, img.shape[-1])).shape,
          img[mask == GC_PR_BGD].reshape((-1, img.shape[-1])).shape)
    print(img[mask == GC_FGD].reshape((-1, img.shape[-1])).shape,
          img[mask == GC_PR_FGD].reshape((-1, img.shape[-1])).shape)

    bgGMM.fit(bg_pixels)
    fgGMM.fit(fg_pr_pixels)

    return bgGMM, fgGMM


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
    return beta


# question 2.3 - Amir
def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]

    fg_D = - fgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_D = - bgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])


    # closure function to calculate boundary energy
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

    # BUILD GRAPH
    num_pix = img.shape[0] * img.shape[1]

    def vid(i, j):  # vertex ID
        return (img.shape[1] * i) + j

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
                weights.append(lamda)
            else:
                edges.append((vid(i, j), S))
                weights.append(bg_D[i, j])

                edges.append((vid(i, j), T))
                weights.append(fg_D[i, j])

            # add edges to neighbours
            add_neighbors(i, j, edges, weights)

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
            new_mask[ind(v)] = GC_PR_FGD

    return new_mask


# question 2.5 - Amir - Done
def check_convergence(energy):
    threshold = 1
    return energy < threshold


# question 2.6 - Aviv
def cal_metric(predicted_mask, gt_mask):
    intersection = np.logical_and(predicted_mask, gt_mask)
    union = np.logical_or(predicted_mask, gt_mask)
    jaccard = np.sum(intersection) / np.sum(union)
    accuracy = np.sum(predicted_mask == gt_mask) / gt_mask.size
    return accuracy, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
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
