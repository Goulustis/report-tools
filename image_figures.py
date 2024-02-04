import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

SAVE_DIR="tmp"

class ImagePointSelector:
    def __init__(self, image_array):
        self.image_array = image_array
        self.points = []

    def select_points(self):
        # Display the image
        plt.imshow(self.image_array)
        plt.title("Click on two points in the image")

        # Define the event handler for mouse clicks
        def onclick(event):
            # Record the point and plot it
            self.points.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()

            # If two points are selected, close the plot
            if len(self.points) == 2:
                plt.close()

        # Connect the event handler and show the plot
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        # Return the selected points
        # self.points = [(int(x), int(y)) for (x,y) in self.points]
        self.points = [(int(y), int(x)) for (x, y) in self.points]
        return self.points

def crop(image, point1, point2):
    """
    Crop an image based on two points.

    :param image: A NumPy array representing the image.
    :param point1: A tuple (x1, y1) representing the first point.
    :param point2: A tuple (x2, y2) representing the second point.
    :return: Cropped image as a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array")

    h1, w1 = point1
    h2, w2 = point2

    # Ensure coordinates are within the image dimensions and correct for slicing
    h_st, h_end = min(h1, h2), max(h1, h2)
    w_st, w_end = min(w1, w2), max(w1, w2)

    return image[h_st:h_end, w_st:w_end]


def load_img(img_f):
    img = plt.imread(img_f)
    h, w, c = img.shape
    hw = w//2

    return img[:,:hw], img[:, hw:]

def load_simple(img_f):
    img = plt.imread(img_f)

    return None, img


def main(*args, load_fn = load_img, use_saved_pnts=False):
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    gt, _ = load_fn(args[0])
    preds = [load_fn(f)[1] for f in args]

    pnts_f = osp.join(SAVE_DIR, "pnts.npy")
    if use_saved_pnts:
        pnts = np.load(pnts_f)
    else:
        selector = ImagePointSelector(preds[0])
        pnts = selector.select_points()
        np.save(pnts_f, np.array(pnts))
    
    crop_fn = lambda x : crop(x, pnts[0], pnts[1])


    if gt is not None:
        gt_crop = crop_fn(gt)
        pr_crops = list(map(crop_fn, preds))
        comb_img = np.concatenate([gt_crop] + pr_crops, axis=1)
    else:
        pr_crops = list(map(crop_fn, preds))
        comb_img = np.concatenate(pr_crops, axis=1)


    plt.imsave("comb_img.png", comb_img)
        
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(gt_crop)
    # axs[0].set_title("gt")

    # axs[1].imshow(pr_crops[0])
    # axs[1].set_title("rgb only")

    # plt.savefig("triangle.png")


if __name__ == "__main__":

    img_fs = [
            "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/outputs/black_seoul_b3_v3/double_constraint_powpow/2024-01-23_020637/eval_results/img/006.png"
            ]



    main(*img_fs, load_fn=load_img, use_saved_pnts=False)
