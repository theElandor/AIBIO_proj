import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
import math
from typing import Any
from torchvision.io import decode_image
from bio_utils import min_max_scale
import utils
import numpy as np
class DataAugmentationDINO_easy(object):
    """
    Simplified augmentations for an easier learning task. Improves stability.
    """
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            utils.GaussianBlur(1.0),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            utils.GaussianBlur(0.1),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            utils.GaussianBlur(p=0.5),
        ])

    def __call__(self, images):
        # images: (B,2,C,H,W)
        crops = []
        im1 = images[:,0,:,:,:].squeeze(1)
        im2 = images[:,1,:,:,:].squeeze(1)
        crops.append(self.global_transfo1(im1).float())
        crops.append(self.global_transfo2(im1).float())
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(im2).float())
        return crops

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomApply(
                 [utils.Wrapper6C(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1))],
                 p=0.8
             ),
            utils.Wrapper6C(transforms.RandomGrayscale(p=0.2)),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Wrapper6C(transforms.RandomSolarize(128, p=0.2))
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
        ])

    def __call__(self, images):
        # images: (B,2,C,H,W)
        crops = []
        im1 = images[:,0,:,:,:].squeeze(1)
        im2 = images[:,1,:,:,:].squeeze(1)
        crops.append(self.global_transfo1(im1).float())
        crops.append(self.global_transfo2(im1).float())
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(im2).float())
        return crops
    
def channelnorm_collate(batch): 
    '''
    Collate function for supervised training
    It performs channel-wise normalization
    '''
    paths, sirna_ids, metadatas = zip(*batch)

    #paths dimensionality: (batch_size,2,num_paths)
    #sirna_ids dimensionality: (batch_size,2)
    #metadatas dimensionality: (batch_size,2,len(metadata_list))
    paths_1, _ = zip(*paths)
    sirna_id_1 , _ = zip(*sirna_ids)
    metadata_1, _ = zip(*metadatas)
    
    norm_images = []    
    for i, path_tuple in enumerate(paths_1):

        stacked_image = torch.cat([F.resize(decode_image(path),224) for path in path_tuple],dim=0).to(torch.float16) #stacking of the channels
        #getting the tuples out of the metadata
        mean_tuple = [float(x) for x in metadata_1[i][-2].strip("()").split(",")]
        variance_tuple = [float(x) for x in metadata_1[i][-1].strip("()").split(",")]
        
        std_tuple = tuple(math.sqrt(element) for element in variance_tuple)
        
        stacked_image_norm = F.normalize(inpt=stacked_image,
                                         mean = list(mean_tuple),
                                         std = list(std_tuple)
                                         )
        norm_images.append(stacked_image_norm)
        
    norm_images = torch.stack(norm_images)
    return norm_images, sirna_id_1, metadata_1


def tuple_channelnorm_collate(batch):
    return tuple_collate(batch, DataAugmentationDINO)

def tuple_channelnorm_collate_easy(batch):
    return tuple_collate(batch, DataAugmentationDINO_easy)

def tuple_collate(batch, augmentation):
    '''
    Collate function for supervised training
    It performs channel-wise normalization
    '''
    paths, sirna_ids, metadatas = zip(*batch)
    # hardcoded for simplicity
    transform = augmentation(
        (0.4, 1.), 
        (0.05, 0.4),
        8,
    )
    #paths dimensionality: (batch_size,2,num_paths)
    #sirna_ids dimensionality: (batch_size,2) tuple
    #metadatas dimensionality: (batch_size,2,len(metadata_list)) tuple
    paths_1, paths_2 = zip(*paths)
    sirna_id_1 , sirna_id_2 = zip(*sirna_ids)
    metadata_1, metadata_2 = zip(*metadatas)
    images = [] # stores non normalized images

    means = [] # store means
    stds = [] # store stds
    for i, (path_tuple_1, path_tuple_2) in enumerate(zip(paths_1, paths_2)):
        decoded_images_1 = [F.resize(decode_image(path), 224) for path in path_tuple_1]
        decoded_images_2 = [F.resize(decode_image(path), 224) for path in path_tuple_2]

        stacked_image_1 = torch.cat(decoded_images_1, dim=0)
        stacked_image_2 = torch.cat(decoded_images_2, dim=0)
        
        mean_tuple_1 = [float(x) for x in metadata_1[i][-2].strip("()").split(",")]
        std_tuple_1 = [math.sqrt(float(x)) for x in metadata_1[i][-1].strip("()").split(",")]        

        mean_tuple_2 = [float(x) for x in metadata_2[i][-2].strip("()").split(",")]
        std_tuple_2 = [math.sqrt(float(x)) for x in metadata_2[i][-1].strip("()").split(",")]

        m12 = torch.stack([torch.tensor(mean_tuple_1), torch.tensor(mean_tuple_2)], dim=0)
        s12 = torch.stack([torch.tensor(std_tuple_1), torch.tensor(std_tuple_2)], dim=0)
        means.append(m12)
        stds.append(s12)

        images.append(torch.stack([stacked_image_1, stacked_image_2], dim=0))

    images = torch.stack(images,dim=0)  #dimensionality: (batch_size,2,n_channels,w,h)
    # ================ APPLY DINO AUGMENTATIONS + NORMALIZE ================
    crops = transform(images)
    for i,(m, s) in enumerate(zip(means, stds)):
        # m.shape -> (2, 6)
        # s.shape -> (2, 6)
        # global crops normalizations
        n1 = transforms.Normalize(mean=m[0,:], std=s[0,:])
        n2 = transforms.Normalize(mean=m[1,:], std=s[1,:])

        crops[0][i,:,:,:] = n1(crops[0][i,:,:,:])
        crops[1][i,:,:,:] = n1(crops[1][i,:,:,:])
        # local crops normalizations
        for j in range(2,10):            
            crops[j][i,:,:,:] = n2(crops[j][i,:,:,:])
    # fix metadata to (2,13,B) shape instead of (B,2,13)
    # WARNING: sirna still needs to be fixed
    m = np.array(metadatas)
    B = m.shape[0]
    V = m.shape[1]
    M = m.shape[2]
    n = np.empty((V,M,B),dtype=object)
    for i in range(B):
        for j in range(V):
            n[j,:,i] = m[i,j,:]
    return crops, sirna_ids, n.tolist()

