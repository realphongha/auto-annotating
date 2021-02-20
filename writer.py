from abc import ABC, abstractmethod


class LabelWriter(ABC):
    def __init__(self, file_path, dets):
        self.file_path = file_path
        self.dets = dets

    @abstractmethod
    def save(self):
        pass