import cv2

def crop_image(image_path, xmin, ymin, xmax, ymax):
    image = cv2.imread(image_path)
    xmin = max(0, image.shape[1] * xmin)
    xmax = max(0, image.shape[1] * xmax)
    ymin = max(0, image.shape[0] * ymin)
    ymax = max(0, image.shape[0] * ymax)

    image_ = image[int(ymin): int(ymax), int(xmin): int(xmax)]
    
    return(image_)
