from writer import LabelWriter
from pascal_voc_writer import Writer

# COCO classes:
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


class PascalVOCWriter(LabelWriter):
    def __init__(self, file_path, dets, img_path, chosen_classes, width, height):
        super().__init__(file_path, dets)
        self.havePerson = False
        self.writer = Writer(img_path, width, height)
        for det in dets:
            class_name = classes[round(det[5])]
            if class_name in chosen_classes:
                self.writer.addObject(class_name, round(det[0]-det[2]/2),
                                      round(det[1]-det[3]/2), round(det[0]+det[2]/2), round(det[1]+det[3]/2))
                self.havePerson = True

    def save(self):
        if self.havePerson:
            self.writer.save(self.file_path)
