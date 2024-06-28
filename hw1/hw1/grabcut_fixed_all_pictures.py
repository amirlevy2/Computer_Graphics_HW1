import numpy as np
import cv2
import argparse
from grabcut_fixed import grabcut, cal_metric



def parse(picture):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default=picture, help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    file_names = ["banana1", "banana2", "book","bush","cross","flower","fullmoon", "grave","llama", "memorial", "sheep", "stone2","teddy"]
    for file_name in file_names:
        try:
            print(file_name)
            args = parse(file_name)

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
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)

