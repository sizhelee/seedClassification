import cv2
import numpy as np

def horizontal_flip(image, axis):
    # flip the image with probablity of 50%, axis=0 or 1
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop ==0:
        image = cv2.flip(image, axis)
    
    return image
    
def random_crop_and_flip(batch_data, padding_size):
    # random crop and flip the image
    flipped_batch = np.zeros_like(batch_data)
    cropped_batch = np.zeros_like(batch_data)
    img_width, img_height = batch_data.shape[2], batch_data.shape[1]
    
    for i in range(len(batch_data)):
        axis = np.random.randint(low=0, high=2)
        flipped_batch[i] = horizontal_flip(image=batch_data[i], axis=axis)

        x_offset = np.random.randint(low=0, high=2*padding_size)
        y_offset = np.random.randint(low=0, high=2*padding_size)
        # print(x_offset, y_offset, img_width, img_height)
        cropped_batch[i][x_offset:x_offset+img_height, 
            y_offset:y_offset+img_width, :] = flipped_batch[i][x_offset:x_offset+img_height, 
            y_offset:y_offset+img_width, :]
        
    return cropped_batch


def normalize_img(image):
    shape = image.shape
    img_normalize = image.reshape(shape[0], -1, shape[3])
    mean = np.expand_dims(np.mean(img_normalize, axis=1), axis=1)
    var = np.expand_dims(np.mean(np.square(img_normalize-mean), axis=1), axis=1)

    img_normalize = (img_normalize - mean)/np.sqrt(var)
    img_normalize = img_normalize.reshape(shape)

    return img_normalize


def deblur_img(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst