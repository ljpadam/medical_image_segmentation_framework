import numpy as np
import torch


class RandomCropSegmentation3D(object):
    def __init__(self, output_size, containLeision = 0):
        assert isinstance(output_size, (int, tuple, list, np.ndarray))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        self.containLeision = containLeision
        self.flipContainLeision = False # used for containLeision = 2, to produce the probability of 0.5
    
    def __call__(self, tempimage, tempGT, name, leisionPositions):
        '''containLeision: 0: must contain lesion
                           1: under the probability of 0.5 contain lesion
                           2: totally random'''
        containLeision = self.containLeision
        if containLeision == 1:
            if self.flipContainLeision:
                containLeision = 0
            else:
                containLeision = 2
            self.flipContainLeision = not self.flipContainLeision

        image_height = tempimage.shape[0]
        image_width = tempimage.shape[1]
        image_depth = tempimage.shape[2]
        findLeision = False
        if containLeision == 0:
            leisionNum = len(leisionPositions)
            leisionIndex = np.random.randint(leisionNum)
            (y, x, z) = leisionPositions[leisionIndex]
            starty = (y - self.output_size[0] * 2//3) + np.random.randint(self.output_size[0]//3)
            startx = (x - self.output_size[1] * 2//3) + np.random.randint(self.output_size[1]//3)
            startz = (z - self.output_size[2] * 2//3) + np.random.randint(self.output_size[2]//3)
            starty = max(starty, 0)
            startx = max(startx, 0)
            startz = max(startz, 0)
            cropimage = tempimage[starty: starty + self.output_size[0], startx: startx + self.output_size[1], startz: startz + self.output_size[2]]
            cropGT = tempGT[starty: starty + self.output_size[0], startx: startx + self.output_size[1], startz: startz + self.output_size[2]]
            cropimage = cropimage.astype(dtype=np.float32)
            cropGT = cropGT.astype(dtype=np.float32)
            
        else:
            starty = np.random.randint(image_height - self.output_size[0])
            startx = np.random.randint(image_width - self.output_size[1])
            startz = np.random.randint(image_depth - self.output_size[2])
            cropimage = tempimage[starty: starty + self.output_size[0], startx: startx + self.output_size[1], startz: startz + self.output_size[2]]
            cropGT = tempGT[starty: starty + self.output_size[0], startx: startx + self.output_size[1], startz: startz + self.output_size[2]]

        if cropimage.shape != (64, 64, 24):
            print startx, stary, startz
            print cropimage.shape
        cropimage = cropimage.astype(dtype=np.float32)
        cropGT = cropGT.astype(dtype=np.float32)
        return cropimage, cropGT

class RandomRotateSegmentation3D(object):
    '''rotate the 3D image in the 2D plane manner, rotate 0 or 90 or 180 degree '''
    def __init__ (self):
        pass
    
    def __call__ (self, img, label, name, leisionPositions):
        randomi = np.random.randint(4)
        cropimage = np.rot90(img, randomi)
        cropGT = np.rot90(label, randomi)
        return img, label

class ToTensorSegmentation(object):
      ''' convert nd img and label to Tensors.'''
      def __init__ (self):
        pass
      def __call__(self, img, label, name, leisionPositions):
          if len(img.shape) == 3:
              img = img.reshape([1,img.shape[0], img.shape[1], img.shape[2]])
              label = label.reshape([1, label.shape[0], label.shape[1], label.shape[2]])
          return torch.from_numpy(img), torch.from_numpy(label).long()


