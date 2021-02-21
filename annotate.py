import torch
import os
import sys
from PIL import Image
from voc_writer import PascalVOCWriter
from multiprocessing.pool import ThreadPool

# yolov5x is the default model to be used
MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# counts labeled images
file_counter = 0


def get_dets(img, model):
    return model(img)


def auto_annotate(img_dir, label_dir, img_ext, classes, save_img, label_writer, 
    model=MODEL):
    global file_counter
    file_counter = 0
    list_img = sorted(os.listdir(img_dir))
    total_file = len(list_img)
    img_paths = [os.path.join(img_dir, name) for name in list_img]
    batch_dets = get_dets(img_paths, model)
    for i in range(len(list_img)):
        img_name = list_img[i]
        if img_name[-3:] not in img_ext:
            continue
        print("Processing %s..." % img_name)
        img_path = img_paths[i]
        xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
        img = Image.open(img_path)
        h, w = img.height, img.width
        c = len(img.getbands())
        result = batch_dets[i]
        dets = [x.numpy() for x in result.xywh]
        writer = label_writer(xml_path, dets, img_path, classes, w, h)
        writer.save()
        if save_img:
            result.save(label_dir)
        file_counter += 1
        print("Done %d/%d" % (file_counter, total_file))


def annotate_single_img(args):
    global file_counter
    i, list_img, img_ext, img_dir, label_dir, model, classes, save_img, label_writer, total_file, img_paths, batch_dets = args
    img_name = list_img[i]
    if img_name[-3:] not in img_ext:
        return
    print("Processing %s..." % img_name)
    img_path = img_paths[i]
    xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
    img = Image.open(img_path)
    h, w = img.height, img.width
    c = len(img.getbands())
    result = batch_dets[i]
    dets = [x.numpy() for x in result.xywh]
    writer = label_writer(xml_path, dets, img_path, classes, w, h)
    writer.save()
    if save_img:
        result.save(label_dir)
    file_counter += 1
    print("Done %d/%d" % (file_counter, total_file))


def auto_annotate_multi_thread(img_dir, label_dir, img_ext, classes, 
    save_img, label_writer, model=MODEL, thread=2):
    if thread < 2:
        print("Pls use more than 1 thread!")
        return
    global file_counter
    file_counter = 0
    list_img = sorted(os.listdir(img_dir))
    total_file = len(list_img)
    img_paths = [os.path.join(img_dir, name) for name in list_img]
    batch_dets = get_dets(img_paths, model)
    with ThreadPool(thread) as pool:
        for _ in pool.map(annotate_single_img, [(i, list_img, img_ext, img_dir, label_dir, model, classes, save_img, label_writer, total_file, img_paths, batch_dets)
                                                for i in range(len(list_img))]):
            pass


if __name__ == "__main__":
    IMG_DIR = r"/" # data directory
    IMG_EXT = ["jpg"] # acceptable image extensions
    LABEL_DIR = r"/" # path to save labels
    CLASSES = ["person"] # classes to be labeled
    SAVE_IMAGE = False # save image or not? (saving takes a lot of time)
    SAVE_TYPE = 'voc' # label type
    MULTI_THREAD = False # use multi thread or not?
    NUM_THREAD = 4 # number of threads
    if len(sys.argv) >= 7:
        IMG_DIR = sys.argv[1].strip()
        IMG_EXT = [ext for ext in sys.argv[2].split(",")]
        LABEL_DIR = sys.argv[3]
        CLASSES = [class_name for class_name in sys.argv[4].split(",")]
        SAVE_IMAGE = (True if sys.argv[5] == "True" else False)
        SAVE_TYPE = sys.argv[6]
        if len(sys.argv) == 9:
            MULTI_THREAD = (True if sys.argv[7] == "True" else False)
            NUM_THREAD = int(sys.argv[8])

    print("Config:", IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, 
            SAVE_IMAGE, SAVE_TYPE, MULTI_THREAD, NUM_THREAD)

    # only supports PascalVOC for now
    if SAVE_TYPE == 'voc':
        label_writer = PascalVOCWriter

    if MULTI_THREAD:
        auto_annotate_multi_thread(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, 
            SAVE_IMAGE, label_writer, thread=NUM_THREAD)
    else:
        auto_annotate(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, SAVE_IMAGE, 
            label_writer)
