import random
import numpy as np

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std. Default: imagenet normalization"""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(img.shape[2]):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img


def random_fliplr(*arrays):
    if random.random() > 0.5:   
        arrays = tuple([np.fliplr(array) for array in arrays])
    
    return arrays

def random_flipud(*arrays):
    if random.random() > 0.5:   
        arrays = tuple([np.flipud(array) for array in arrays])
    
    return arrays

def random_rot(*arrays):
    k = random.randrange(3) + 1
    arrays = tuple([np.rot90(array, k).copy() for array in arrays])
    
    return arrays

def random_crop(*images, crop_size, mean=None, ignore_index=255,  **labels):
    if mean is None:
        mean = [0] * sum([img.shape[2] for img in images])

    h, w = list(labels.values())[0].shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_images = []
    pad_labels = {}

    for idx, img in enumerate(images):
        # if len(img.shape) != 3:
        #     print(idx)
        pad_image = np.zeros((H, W, img.shape[2]), dtype=np.float32)
        mean_idx = 0
        for i in range(img.shape[2]):
            pad_image[:, :, i] = mean[mean_idx]
            mean_idx += 1
        pad_images.append(pad_image)

    for key, label in labels.items():
        pad_labels[key] = np.ones((H, W), dtype=np.float32) * ignore_index

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    for idx, img in enumerate(images):
        pad_images[idx][H_pad:H_pad + h, W_pad:W_pad + w, :] = img

    for key, label in labels.items():
        pad_labels[key][H_pad:H_pad + h, W_pad:W_pad + w] = label

    def get_random_cropbox(cat_max_ratio=0.75):
        for _ in range(10):
            H_start = random.randrange(0, H - crop_size + 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1)
            W_end = W_start + crop_size

            temp_label = pad_labels[list(labels.keys())[0]][H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()

    cropped_images = [img[H_start:H_end, W_start:W_end, :] for img in pad_images]
    cropped_labels = {key: label[H_start:H_end, W_start:W_end] for key, label in pad_labels.items()}

    return (*cropped_images, *cropped_labels.values())