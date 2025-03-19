import torch
import torchvision.transforms.v2.functional as F
import math
from torchvision.io import decode_image

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

def tuple_channelnorm_collate_dino(batch):
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
    return norm_images, torch.tensor(sirna_ids)[:,0].tolist(), metadatas

def tuple_channelnorm_collate_head(batch):
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
        #decoded_images_2 = [F.resize(decode_image(path), 224) for path in path_tuple_2]

        stacked_image_1 = torch.cat(decoded_images_1, dim=0).to(torch.float32)
        #stacked_image_2 = torch.cat(decoded_images_2, dim=0).to(torch.float32)

        mean_tuple_1 = eval(metadata_1[i][-2])
        variance_tuple_1 = eval(metadata_1[i][-1])
        std_tuple_1 = tuple(math.sqrt(x) for x in variance_tuple_1)
        
        # mean_tuple_2 = eval(metadata_2[i][-2])
        # variance_tuple_2 = eval(metadata_2[i][-1])
        # std_tuple_2 = tuple(math.sqrt(x) for x in variance_tuple_2)

        stacked_image_norm_1 = F.normalize(stacked_image_1, mean=list(mean_tuple_1), std=list(std_tuple_1))
        #stacked_image_norm_2 = F.normalize(stacked_image_2, mean=list(mean_tuple_2), std=list(std_tuple_2))

        norm_images.append(stacked_image_norm_1)

    norm_images = torch.stack(norm_images,dim=0)  #dimensionality: (batch_size,2,n_channels,w,h)
    return norm_images, torch.tensor(sirna_ids)[:,0].tolist(), metadata_1