import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
import math
from typing import Tuple,List,Any
from torchvision.io import decode_image
import time
from source.utils import min_max_scale
'''def simclr_collate(batch):
    """
    Custom collate function for processing a batch of Rxrx1 dataset images.

    This function performs the following steps:
    1. Converts images from uint8 to float format.
    2. Normalizes images using mean and variance from metadata.
    3. Applies a series of augmentations to each normalized image.
    4. Stacks and concatenates original and augmented images.

    The batch will be composed by:
     - batch_size normal images
     - batch size augmented images
    The correspondance between each original image and its augmented counterpart is:
    batch[i] --> batch[i + 256]

    Args:
        batch (list of tuples): Each tuple contains:
            - image (Tensor): Raw image in uint8 format.
            - sirna_id (int): Identifier for the sirna.
            - metadata (list): Metadata containing mean and variance for normalization.

    Returns:
        tuple: (tot_images, sirna_ids, metadata)
            - tot_images (Tensor): Concatenated tensor of normalized and augmented images.
            - sirna_ids (tuple): Tuple of sirna IDs.
            - metadata (tuple): Tuple of metadata for each sample.
    """
    image_to_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float)])
    # view for self supervised learning
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0))
])
    images, sirna_ids, metadata = zip(*batch)
    augmented_images = []
    norm_images = []
    for i, image in enumerate(images):
        mean = metadata[i][11]
        variance = metadata[i][12]
        image = image_to_tensor(image)
        image = (image - mean)/math.sqrt(variance)
        aug_image = augmentation(image)

        augmented_images.append(aug_image)
        norm_images.append(image)

    norm_images = torch.stack(norm_images) 
    augmented_images = torch.stack(augmented_images)

    tot_images = torch.cat([norm_images,augmented_images],dim=0)

    return tot_images, sirna_ids, metadata'''

'''def simple_collate(batch):
    """
    Simple collate function used to normalize images for supervised learning.
    It performs the same steps of the simclr_collate without augmentation,
    so it only performs normalization.
    """
    image_to_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float)])    
    images, sirna_ids, metadata = zip(*batch)    
    norm_images = []
    for i, image in enumerate(images):
        mean = metadata[i][11]
        variance = metadata[i][12]
        image = image_to_tensor(image)
        image = (image - mean)/math.sqrt(variance)
        norm_images.append(image)
    norm_images = torch.stack(norm_images)         
    return norm_images, sirna_ids, metadata'''

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

        stacked_image = torch.cat([F.resize(decode_image(path),224) for path in path_tuple],dim=0).to(torch.float32) #stacking of the channels
        #getting the tuples out of the metadata
        mean_tuple = eval(metadata_1[i][-2])
        variance_tuple = eval(metadata_1[i][-1])
        
        std_tuple = tuple(math.sqrt(element) for element in variance_tuple)
        
        stacked_image_norm = F.normalize(inpt=stacked_image,
                                         mean = list(mean_tuple),
                                         std = list(std_tuple)
                                         )
        norm_images.append(stacked_image_norm)
        
    norm_images = torch.stack(norm_images)
    return norm_images, sirna_id_1, metadata_1

def tuple_channelnorm_collate(batch):
    '''
    Collate function for supervised training
    It performs channel-wise normalization
    '''
    paths, sirna_ids, metadatas = zip(*batch)

    #paths dimensionality: (batch_size,2,num_paths)
    #sirna_ids dimensionality: (batch_size,2) tuple
    #metadatas dimensionality: (batch_size,2,len(metadata_list)) tuple
    paths_1, paths_2 = zip(*paths)
    sirna_id_1 , sirna_id_2 = zip(*sirna_ids)
    metadata_1, metadata_2 = zip(*metadatas)
    norm_images = []
    
    for i, (path_tuple_1, path_tuple_2) in enumerate(zip(paths_1, paths_2)):
        decoded_images_1 = [F.resize(decode_image(path), 224) for path in path_tuple_1]
        decoded_images_2 = [F.resize(decode_image(path), 224) for path in path_tuple_2]

        stacked_image_1 = torch.cat(decoded_images_1, dim=0).to(torch.float32)
        stacked_image_2 = torch.cat(decoded_images_2, dim=0).to(torch.float32)

        mean_tuple_1 = eval(metadata_1[i][-2])
        variance_tuple_1 = eval(metadata_1[i][-1])
        std_tuple_1 = tuple(math.sqrt(x) for x in variance_tuple_1)
        
        mean_tuple_2 = eval(metadata_2[i][-2])
        variance_tuple_2 = eval(metadata_2[i][-1])
        std_tuple_2 = tuple(math.sqrt(x) for x in variance_tuple_2)

        stacked_image_norm_1 = F.normalize(stacked_image_1, mean=list(mean_tuple_1), std=list(std_tuple_1))
        stacked_image_norm_2 = F.normalize(stacked_image_2, mean=list(mean_tuple_2), std=list(std_tuple_2))

        norm_images.append(torch.stack([stacked_image_norm_1, stacked_image_norm_2], dim=0))

    norm_images = torch.stack(norm_images,dim=0)  #dimensionality: (batch_size,2,n_channels,w,h)
    #OUTPUT SHAPES:
    #norm images --> #TODO
    return norm_images, sirna_ids, metadatas 

'''def dino_test_collate(batch):
    #TODO METTERLA A POSTO
    """Transformations used for dino training that must be used also 
    during validation and head training. For now we are just
    normalizing using ImageNet values of mean and standard deviation.
    A little bit of code is repeated here for the sake of testing.
    Requires 3 channel metadata."""
    
    crop = transforms.Compose([
        transforms.CenterCrop(224)
        ])
    images, sirna_ids, metadata = zip(*batch)
    
    norm_images = []
    for i, image in enumerate(images):
        
        mean_tuple = eval(metadata[i][11])
        variance_tuple = eval(metadata[i][12])
        
        #converting the tuples to tensors
        mean_tensor = (torch.tensor(mean_tuple).view(3,1,1))/255.0
        std_tensor = (torch.sqrt(torch.tensor(variance_tuple)).view(3,1,1))/255.0

        image = image.float() / 255.0 #convert to 0-1 range
        image = (image - mean_tensor)/std_tensor # shift using ImageNet mean and std
        image = crop(image) # center crop according to ViT standard input
        norm_images.append(image)
    norm_images = torch.stack(norm_images)         
    return norm_images, sirna_ids, metadata'''
