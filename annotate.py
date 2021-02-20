import torch
import os
import sys
from PIL import Image
from voc_writer import PascalVOCWriter
from multiprocessing.pool import ThreadPool

MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)


def get_dets(img, model):
    return model(img)


def auto_annotate(img_dir, label_dir, img_ext, classes, save_img, label_writer, 
    model=MODEL):
    for img_name in os.listdir(img_dir):
        if img_name[-3:] not in img_ext:
            continue
        print("Processing %s..." % img_name)
        img_path = os.path.join(img_dir, img_name)
        xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
        img = Image.open(img_path)
        h, w = img.height, img.width
        c = len(img.getbands())
        result = get_dets(img, model)
        dets = [x.numpy() for x in result.xywh[0]]
        writer = label_writer(xml_path, dets, img_path, classes, w, h)
        writer.save()
        if save_img:
            result.save(label_dir)


def annotate_single_img(args):
    img_name, img_ext, img_dir, label_dir, model, classes, save_img, label_writer = args
    if img_name[-3:] not in img_ext:
        return
    print("Processing %s..." % img_name)
    img_path = os.path.join(img_dir, img_name)
    xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
    img = Image.open(img_path)
    h, w = img.height, img.width
    c = len(img.getbands())
    result = get_dets(img, model)
    dets = [x.numpy() for x in result.xywh[0]]
    writer = label_writer(xml_path, dets, img_path, classes, w, h)
    writer.save()
    if save_img:
        result.save(label_dir)


def auto_annotate_multi_thread(img_dir, label_dir, img_ext, classes, 
    save_img, label_writer, model=MODEL, thread=2):
    if thread < 2:
        print("Pls use more than 1 thread!")
        return
    with ThreadPool(thread) as pool:
        for _ in pool.map(annotate_single_img, [(img_name, img_ext, img_dir, label_dir, model, classes, save_img, label_writer)
                                                for img_name in os.listdir(img_dir)]):
            pass


if __name__ == "__main__":
    print(sys.argv)
    IMG_DIR = r"/"
    IMG_EXT = ["jpg"]
    LABEL_DIR = r"/"
    CLASSES = ["person"]
    SAVE_IMAGE = False
    SAVE_TYPE = 'voc'
    MULTI_THREAD = True
    NUM_THREAD = 4
    if len(sys.argv) >= 7:
        IMG_DIR = sys.argv[1].strip()
        IMG_EXT = [ext for ext in sys.argv[2].split(",")]
        LABEL_DIR = sys.argv[3]
        CLASSES = [class_name for class_name in sys.argv[4].split(",")]
        SAVE_IMAGE = bool(sys.argv[5])
        SAVE_TYPE = sys.argv[6]
        if len(sys.argv) == 9:
            MULTI_THREAD = bool(sys.argv[7])
            NUM_THREAD = int(sys.argv[8])

    if SAVE_TYPE == 'voc':
        label_writer = PascalVOCWriter

    if MULTI_THREAD:
        auto_annotate_multi_thread(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, 
            SAVE_IMAGE, label_writer, thread=NUM_THREAD)
    else:
        auto_annotate(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, SAVE_IMAGE, 
            label_writer)
