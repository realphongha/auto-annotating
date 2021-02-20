import torch
import os
from PIL import Image
from voc_writer import PascalVOCWriter
from multiprocessing.pool import ThreadPool

MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)


def auto_annotate(img_dir, label_dir, img_ext, classes, model=MODEL):
    for img_name in os.listdir(img_dir):
        if img_name[-3:] not in img_ext:
            continue
        print("Processing %s..." % img_name)
        img_path = os.path.join(img_dir, img_name)
        xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
        img = Image.open(img_path)
        h, w = img.height, img.width
        c = len(img.getbands())
        result = model(img)
        dets = [x.numpy() for x in result.xywh[0]]
        writer = PascalVOCWriter(xml_path, dets, img_path, classes, w, h)
        writer.save()
        result.save(label_dir)


def annotate_single_img(args):
    img_name, img_ext, img_dir, label_dir, model, classes = args
    if img_name[-3:] not in img_ext:
        return
    print("Processing %s..." % img_name)
    img_path = os.path.join(img_dir, img_name)
    xml_path = os.path.join(label_dir, img_name[:-3] + "xml")
    img = Image.open(img_path)
    h, w = img.height, img.width
    c = len(img.getbands())
    result = model(img)
    dets = [x.numpy() for x in result.xywh[0]]
    writer = PascalVOCWriter(xml_path, dets, img_path, classes, w, h)
    writer.save()
    result.save(label_dir)


def auto_annotate_multi_thread(img_dir, label_dir, img_ext, classes, model=MODEL, thread=2):
    if thread < 2:
        print("Pls use more than 1 thread!")
        return
    with ThreadPool(thread) as pool:
        for _ in pool.map(annotate_single_img, [(img_name, img_ext, img_dir, label_dir, model, classes)
                                                for img_name in os.listdir(img_dir)]):
            pass


if __name__ == "__main__":
    IMG_DIR = r"E:\Workspace\data\prj15\phong\data"
    IMG_EXT = ["jpg"]
    LABEL_DIR = r"E:\Workspace\data\prj15\label\box_camera_2_2021_02_04_13_29_34"
    CLASSES = ["person"]

    # auto_annotate(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES)
    auto_annotate_multi_thread(IMG_DIR, LABEL_DIR, IMG_EXT, CLASSES, thread=2)
