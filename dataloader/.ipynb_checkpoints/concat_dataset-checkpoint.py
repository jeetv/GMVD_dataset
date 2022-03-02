import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
#from MCMT.resnet import resnet18
import matplotlib.pyplot as plt
#import torch.utils.data.ConcatDataset as cd

class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *datasets):
        super(ConcatDataset, self).__init__(datasets)
        #self.datasets = datasets
        self.dataset_list = list(d.root for d in datasets)#list(d.root.split('/')[-1] for d in datasets)
        print(self.dataset_list)
        self.dicts = {}
        for i,d in enumerate(self.datasets):
            self.config_dict = {}
            self.config_dict['base'] = d
            self.config_dict['num_cam'] = d.num_cam
            self.config_dict['img_shape'] = d.img_shape
            self.config_dict['reducedgrid_shape'] = d.reducedgrid_shape
            self.config_dict['img_reduce'] = d.img_reduce
            
            upsample_shape = list(map(lambda x: int(x / d.img_reduce), d.img_shape))
            self.config_dict['upsample_shape'] = upsample_shape
            img_reduce = np.array(d.img_shape) / np.array(upsample_shape)
            img_zoom_mat = np.diag(np.append(img_reduce, [1]))
            self.config_dict['img_zoom_mat'] = img_zoom_mat
            map_zoom_mat = np.diag(np.append(np.ones([2]) / d.grid_reduce, [1]))
            self.config_dict['map_zoom_mat'] = map_zoom_mat
            
            imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(d.num_cam, d.base.intrinsic_matrices,
                                                                           d.base.extrinsic_matrices,
                                                                           d.base.worldgrid2worldcoord_mat)
            self.config_dict['imgcoord2worldgrid_matrices'] = imgcoord2worldgrid_matrices
            
            # Projection matrix
            self.config_dict['proj_mats'] = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                              for cam in range(d.num_cam)]
            
            # Coordinate Map
            self.config_dict['coord_map'] = self.get_coord_map(d.reducedgrid_shape + [1])
            
            self.dicts[self.dataset_list[i]] = self.config_dict 
            
        #print(list(d.root.split('./')[1] for d in self.datasets))
      
    '''
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
    '''

    def __len__(self):
        return sum(len(d) for d in self.datasets)
    
    def get_imgcoord2worldgrid_matrices(self, num_cam, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(num_cam):
            # removing third column(z=0) from extrinsic matrix of size 3x4 
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)

            # transforming img axis to grid map axis
            # x(col),y(row) img coord -->  y(col), x(row) grid map coord
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices
    
    def get_coord_map(self, grid_shape):
        H, W, C = grid_shape
        # img[y,x] = img[h,w]
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        # making x and y in range [-1.0 to 1.0]
        grid_x = torch.from_numpy((grid_x / (W - 1) * 2) - 1).float()
        grid_y = torch.from_numpy((grid_y / (H - 1) * 2) - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return ret