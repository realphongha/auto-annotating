import torch
import os
import sys
from PIL import Image
from voc_writer import PascalVOCWriter
from multiprocessing.pool import ThreadPool

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.cuda.current_device()

# yolov5x is the default model to be used
MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).to(device)


# counts labeled images
file_counter = 0


def get_dets(imgs, model, batch_size):
    # batch predicting
    dets = list()
    data_size = len(imgs)
    for i in range(data_size//batch_size + 1):
        print("Reading %d/%d..." % (i*batch_size+1, data_size))
        if i < data_size//batch_size:
            result = model(imgs[i*batch_size:(i+1)*batch_size])
            for img in imgs[i*batch_size:(i+1)*batch_size]:
                img.close()
        else:
            result = model(imgs[i*batch_size:])
            for img in imgs[i*batch_size:]:
                img.close()
        result = [x.cpu().numpy() for x in result.xywh]
        dets.extend(result)
    return dets


def auto_annotate(img_dir, label_dir, img_ext, classes, save_img, batch_size, 
    label_writer, model=MODEL):
    global file_counter
    file_counter = 0
    list_img = sorted(os.listdir(img_dir))
    total_file = len(list_img)
    img_paths = [os.path.join(img_dir, name) for name in list_img]
    imgs = [Image.open(path) for path in img_paths]
    print("Reading batch images...")
    batch_dets = get_dets(imgs, model, batch_size)
    for i in range(len(list_img)):
        img_name = list_img[i]
        if img_name[-3:] not in img_ext:
            continue
        print("Processing %s..." % img_name)
        img_path = img_paths[i]
        xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
        img = imgs[i]
        h, w = img.height, img.width
        c = len(img.getbands())
        dets = batch_dets[i]
        writer = label_writer(xml_path, dets, img_path, classes, w, h)
        writer.save()
        if save_img:
            result.save(label_dir)
        file_counter += 1
        print("Done %d/%d" % (file_counter, total_file))


def annotate_single_img(args):
    global file_counter
    i, list_img, img_ext, img_dir, label_dir, model, classes, save_img, label_writer, total_file, img_paths, imgs, batch_dets = args
    img_name = list_img[i]
    if img_name[-3:] not in img_ext:
        return
    print("Processing %s..." % img_name)
    img_path = img_paths[i]
    xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
    img = imgs[i]
    h, w = img.height, img.width
    c = len(img.getbands())
    dets = batch_dets[i]
    writer = label_writer(xml_path, dets, img_path, classes, w, h)
    writer.save()
    if save_img:
        result.save(label_dir)
    file_counter += 1
    print("Done %d/%d" % (file_counter, total_file))


def auto_annotate_multi_thread(img_dir, label_dir, img_ext, classes, 
    save_img, batch_size, label_writer, model=MODEL, thread=4):
    if thread < 2:
        print("Pls use more than 1 thread!")
        return
    global file_counter
    file_counter = 0
    list_img = sorted(os.listdir(img_dir))
    total_file = len(list_img)
    img_paths = [os.path.join(img_dir, name) for name in list_img]
    imgs = [Image.open(path) for path in img_paths]
    print("Reading batch images...")
    batch_dets = get_dets(img_paths, model, batch_size)
    with ThreadPool(thread) as pool:
        for _ in pool.map(annotate_single_img, [(i, list_img, img_ext, img_dir, label_dir, model, classes, save_img, label_writer, total_file, img_paths, imgs, batch_dets)
                                                for i in range(len(list_img))]):
            pass


if __name__ == "__main__":
    IMG_DIR = r"/" # data directory
    IMG_EXT = ["jpg"] # acceptable image extensions
    LABEL_DIR = r"/" # path to save labels
    CLASSES = ["person"] # classes to be labeled
    SAVE_IMAGE = False # save image or not? (saving takes a lot of time)
    SAVE_TYPE = 'voc' # label type
    BATCH_SIZE = 100 # batch size for input images
    MULTI_THREAD = False # use multi thread or not?
    NUM_THREAD = 4 # number of threads
    if len(sys.argv) >= 7:
        IMG_DIR = sys.argv[1].strip()
        IMG_EXT = [ext for ext in sys.argv[2].split(",")]
        LABEL_DIR = sys.argv[3]
        CLASSES = [class_name for class_name in sys.argv[4].split(",")]
        SAVE_IMAGE = (True if sys.argv[5] == "True" else False)
        SAVE_TYPE = sys.argv[6]
        BATCH_SIZE = int(sys.argv[7])
        if len(sys.argv) == 9:
            MULTI_THREAD = (True if sys.argv[8] == "True" else False)
            NUM_THREAD = int(sys.argv[9])

    print("Config:", IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, 
            SAVE_IMAGE, SAVE_TYPE, MULTI_THREAD, NUM_THREAD)

    # only supports PascalVOC for now
    if SAVE_TYPE == 'voc':
        label_writer = PascalVOCWriter

    if MULTI_THREAD:
        auto_annotate_multi_thread(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, 
            SAVE_IMAGE, BATCH_SIZE, label_writer, thread=NUM_THREAD)
    else:
        auto_annotate(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, SAVE_IMAGE, 
            BATCH_SIZE, label_writer)
