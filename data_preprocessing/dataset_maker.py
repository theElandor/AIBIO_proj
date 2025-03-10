import sys, os, shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List
import pandas as pd

source_path = '/work/h2020deciderficarra_shared/rxrx1/rxrx1_orig'
destination_path = '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1'

def ask_question(question):
    while True:
        answer = input(f"{question} (Y/N): ").strip().upper()
        if answer in ["Y", "N"]:
            return answer
        print("Invalid input. Please enter 'Y' or 'N'.")
def yes_no_manager(answer):
    if answer == 'Y':
        return
    elif answer == 'N':
        print('OK. Terminating script execution.')
        sys.exit(-1)

        
def clear_previous_line():
    sys.stdout.write("\033[F")  # Move cursor up one line
    sys.stdout.write("\033[K")  # Clear the line
    sys.stdout.flush()
    
def load_six_channel_image(image_paths):
    """Loads six grayscale PNGs and stacks them into two 3-channel images."""
    for path in image_paths:
        assert os.path.exists(path) , f"The path: {path} doesn't exist!"
    channels = [np.array(Image.open(path), dtype=np.uint8) for path in image_paths]
    assert len(image_paths) == 6, f'Got {len(image_paths)} input images instead of 6!'
    return (np.stack(channels[:3], axis=-1) , np.stack(channels[3:],axis=-1))  # Shape: (H, W, 6)

def save_pngs(source_paths:List[str],destination_path):
    '''
    source_paths: a list of 6 paths (six input images)
    destination_paths: the folder in which the .png images will be saved
    '''
    image_012 , image_345 = load_six_channel_image(source_paths)
    base_path = os.path.join(destination_path,source_paths[0][-13:-6])
    
    save_paths_list_012 = [base_path + f'p{i}_c012.png' for i in range(1,6)]
    images_list_012 = [image_012[16:240, 16:240, :],
                  image_012[16:240, 272:496, :],
                  image_012[272:496, 16:240, :],
                  image_012[272:496, 272:496, :],
                  image_012[144:368,144:368,:]
                  ]
    for path,image in zip(save_paths_list_012,images_list_012):
        Image.fromarray(image).save(path)
    
    save_paths_list_345 = [base_path + f'p{i}_c345.png' for i in range(1,6)]
    images_list_345 = [image_345[16:240, 16:240, :],
                  image_345[16:240, 272:496, :],
                  image_345[272:496, 16:240, :],
                  image_345[272:496, 272:496, :],
                  image_345[144:368,144:368, :]
                  ]
    for path,image in zip(save_paths_list_345,images_list_345):
        Image.fromarray(image).save(path)
        
        

def clone_tree(source_path, destination_path, is_first: bool = False):
    """Recursively clones directory structure, parallelizing only the first level."""
    os.makedirs(destination_path, exist_ok=True)  # Ensure root directory exists

    # List all entries in the source directory
    entries = os.listdir(source_path)
    iterator = tqdm(entries, desc='Cloning the tree') if is_first else entries

    # Collect subdirectory tasks
    tasks = []
    for entry in iterator:
        abs_source_entry = os.path.join(source_path, entry)
        abs_destination_entry = os.path.join(destination_path, entry)

        if os.path.isdir(abs_source_entry):
            os.makedirs(abs_destination_entry, exist_ok=True)
            tasks.append((abs_source_entry, abs_destination_entry, False)) 
        else:
            if not abs_source_entry.endswith('1.png'):
                continue
            else:
                all_abs_source_entries = [abs_source_entry[:-5] + f'{i}.png' for i in range(1,7)]
                save_pngs(source_paths=all_abs_source_entries,destination_path=destination_path)

    if is_first and tasks: 
        with Pool(processes=min(len(tasks), cpu_count())) as pool:
            pool.starmap(clone_tree, tasks)  

    else:  
        for task in tasks:
            clone_tree(*task)  
            
  
if __name__ == '__main__':
    #Prompting the user for confirmation before proceeding with data deletion
    #answer = ask_question(f'All the contents of {destination_path} will be erased! Sure you want to continue?')
    #yes_no_manager(answer)
    #clear_previous_line()
    
    #User friendly outputs
    print(f'Source directory: {source_path}',flush=True)
    print(f'Destination directory: {destination_path}',flush=True)
    
    #Creating the destination directory if not alread present
    os.makedirs(destination_path,exist_ok=True)
    
    #Removing all contents of the destination directory
    for entry in tqdm(os.listdir(destination_path),desc = 'Clearing the destination directory'):
        absolute_path_entry = os.path.join(destination_path,entry)
        if os.path.isdir(absolute_path_entry):
            os.system(f'rm -rf {absolute_path_entry}')
        else:
            os.remove(absolute_path_entry)
    print('Finished the clearing!',flush=True)
    #Cloning the source tree
    images_source = os.path.join(source_path,'images')
    images_destination = os.path.join(destination_path,'images')
    metadata_destination = os.path.join(destination_path,'metadata')
    os.makedirs(images_destination)
    os.makedirs(metadata_destination)
    clone_tree(source_path=images_source,destination_path=images_destination,is_first=True)
    print('Done!')
    
    
    
    
    
    