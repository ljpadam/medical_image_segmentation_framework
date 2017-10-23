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
            #print('prepare lesion: ', leisionNum)
            leisionIndex = np.random.randint(leisionNum)
            (y, x, z) = leisionPositions[leisionIndex]
            starty = (y - self.output_size[0] * 2//3) + np.random.randint(self.output_size[0]//3)
            startx = (x - self.output_size[1] * 2//3) + np.random.randint(self.output_size[1]//3)
            startz = (z - self.output_size[2] * 2//3) + np.random.randint(self.output_size[2]//3)
            starty = max(starty, 0)
            startx = max(startx, 0)
            startz = max(startz, 0)
            endy = starty + self.output_size[0]
            endx = startx + self.output_size[1]
            endz = startz + self.output_size[2]
            if endy >= image_height:
                endy = image_height
                starty = endy-self.output_size[0]
            if endx >= image_width:
                endx = image_width
                startx = endx - self.output_size[1]
            if endz >= image_depth:
                endz = image_depth
                startz = endz - self.output_size[2]
            cropimage = tempimage[starty: endy, startx: endx, startz: endz]
            cropGT = tempGT[starty: endy, startx: endx, startz: endz]
            cropimage = cropimage.astype(dtype=np.float32)
            cropGT = cropGT.astype(dtype=np.float32)
            
        else:
            #print ('prepare non-leision: ')
            starty = np.random.randint(image_height - self.output_size[0])
            startx = np.random.randint(image_width - self.output_size[1])
            startz = np.random.randint(image_depth - self.output_size[2])
            cropimage = tempimage[starty: starty + self.output_size[0], startx: startx + self.output_size[1], startz: startz + self.output_size[2]]
            cropGT = tempGT[starty: starty + self.output_size[0], startx: startx + self.output_size[1], startz: startz + self.output_size[2]]

        if not np.all(cropimage.shape == self.output_size):
            print startx, starty, startz
            print 'cropimage size', cropimage.shape
            print 'croplabel size', cropGT.shape
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

class SplitDataAndToTensor(object):
    '''split the full img into many crops according to the stride and crop size,
    the returned data is transformed to tensor, and the batch size is calculated based on the crop size, and the original batch size in dataloader is ignored'''
    def __init___(self, stride, volsize):
        self.stride = stride
        self.volsize = volsize

    def __call__(self, img, label, name, leisionPositions):
        height = int(self.volsize[0])
        width = int(self.volsize[1])
        depth = int(self.volsize[2])
        stride_height = int(self.stride[0])
        stride_width = int(self.stride[1])
        stride_depth = int(self.stride[2])
        whole_height = int(img.shape[0])
        whole_width = int(img.shape[1])
        whole_depth = int(img.shape[2])
        ynum = int(math.ceil((whole_height - height) / float(stride_height))) + 1
        xnum = int(math.ceil((whole_width - width) / float(stride_width))) + 1
        znum = int(math.ceil((whole_depth - depth) / float(stride_depth))) + 1
        imgs = list()
        labels = list()
        coordinates = list()
        # crop the image
        for y in xrange(ynum):
            for x in xrange(xnum):
                for z in xrange(znum):
                    if(y * stride_height + height < whole_height):
                        ystart = y * stride_height
                        yend = ystart + height
                    else:
                        ystart = whole_height - height
                        yend = whole_height
                    if(x * stride_width + width < whole_width):
                        xstart = x * stride_width
                        xend = xstart + width
                    else:
                        xstart = whole_width - width
                        xend = whole_width
                    if(z * stride_depth + depth < whole_depth):
                        zstart = z * stride_depth
                        zend = zstart + depth
                    else:
                        zstart = whole_depth - depth
                        zend = whole_depth
                    imgs.append(img[ystart:yend, xstart:xend, zstart:zend])
                    labels.append(label[ystart:yend, xstart:xend, zstart:zend])
                    coordinates.append(np.asarray([ystart, yend, xstart, xend, zstart, zend]))
        imgs_numpy = np.zeros((len(imgs),1, height, width, depth))
        labels_numpy = np.zeros((len(imgs),1, height, width, depth))
        for i in xrange(len(imgs)):
            imgs_numpy[i,0,:, :, :] = imgs[i]
            labels_numpy[i,0,:, :, :] = labels[i]
        return torch.from_numpy(imgs_numpy), torch.from_numpy(labels_numpy), coordinates
        



