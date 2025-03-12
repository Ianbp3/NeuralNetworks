import numpy as np
import struct

class MnistDataset:
    def __init__(self):
        self.images = None
        self.labels = None

    def load(self, images_filename, labels_filename):
        # Load images
        with open(images_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic}. Expected 2051 for images.")

            buffer = f.read()

            data = np.frombuffer(buffer, dtype=np.uint8)

            self.images = data.reshape(num_images, rows*cols)
            self.images = (self.images > 128).astype(int)


        # Load labels
        with open(labels_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))

            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic}. Expected 2049 for labels.")

            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
            self.onehotlabels = []
            for label in self.labels:
                if label == 0:
                    self.onehotlabels.append([1,0,0,0,0,0,0,0,0,0])
                if label == 1:
                    self.onehotlabels.append([0,1,0,0,0,0,0,0,0,0])
                if label == 2:
                    self.onehotlabels.append([0,0,1,0,0,0,0,0,0,0])
                if label == 3:
                    self.onehotlabels.append([0,0,0,1,0,0,0,0,0,0])
                if label == 4:
                    self.onehotlabels.append([0,0,0,0,1,0,0,0,0,0])
                if label == 5:
                    self.onehotlabels.append([0,0,0,0,0,1,0,0,0,0])
                if label == 6:
                    self.onehotlabels.append([0,0,0,0,0,0,1,0,0,0])
                if label == 7:
                    self.onehotlabels.append([0,0,0,0,0,0,0,1,0,0])
                if label == 8:
                    self.onehotlabels.append([0,0,0,0,0,0,0,0,1,0])
                if label == 9:
                    self.onehotlabels.append([0,0,0,0,0,0,0,0,0,1])