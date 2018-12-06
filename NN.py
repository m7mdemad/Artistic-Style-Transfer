import numpy as np
from commonfunctions import *

def get_patches(img, patch_sizes, subsampling_gaps):
    patches = []
    for i in range(len(patch_sizes)):
        patch = get_patches_aux(img, patch_size=patch_sizes[i], subsampling_gap=subsampling_gaps[i])
        patches.append(patch)
    return patches
    
    
#def get_patches_aux (img, patch_sizes, subsampling_gaps, index):
#    patch_size = patch_sizes[index]
#    gap = subsampling_gaps[index]
#    patches = 
#    vertical_patch_count    = (img.shape[0] - patch_size) // gap + 1
#    horizontal_patch_count   = (img.shape[1] - patch_size) // gap + 1
#    total_patch_count = vertical_patch_count * horizontal_patch_count
#    patches = np.empty((total_patch_count * 4, patch_size * patch_size * 3)) # 4 rotations, 3 channels
#    patch_index = 0
#    for v_index in range(vertical_patch_count):
#        for h_index in range(horizontal_patch_count):
#            patch = img[v_index * gap : v_index * gap + patch_size,
#                        h_index * gap : h_index * gap + patch_size]
#            for i in range(4):                          # consider all rotations of the patch
#                patches[patch_index + i] = np.rot90(patch, k=i).flatten()    # rgb for each pixel one after another
#            patch_index += 4
#            #show_images([patch])
#    return patches
    
from skimage.util import view_as_windows

def get_patches_aux(img, patch_size, subsampling_gap): 
    return view_as_windows(img, window_shape=(patch_size, patch_size, 3), step=subsampling_gap)[:,:,0]

def flatten_patches(patches, rotate = True):
    patches_shape = patches.shape
    patch_count = patches_shape[0] * patches_shape[1]
    pixels_count = patches_shape[2] * patches_shape[3] * patches_shape[4]
    patch_index = 0
    if rotate:
        rotations = 4
    else:
        rotations = 1
        
    flat_patches = np.empty((patch_count * rotations, pixels_count))    # 4 rotations for each patch
    for rotation in range(rotations):
        for row in patches:
            for patch in row:
                flat_patches[patch_index] = np.rot90(patch, k=rotation, axes=(0,1)).ravel()
                patch_index += 1        
    return flat_patches

from sklearn.decomposition import PCA

def apply_pca(patches, pca=None, inverse=False):
    if pca is None:
        pca = PCA(0.95)  # keep only 95% ov the info (variance)
        pca.fit(patches)
        return pca.transform(patches), pca
    
    if inverse:
        return pca.inverse_transform(patches)
    else:
        return pca.transform(patches)
        

from sklearn.preprocessing import StandardScaler

def apply_standard_Scaler(patches, scaler=None, inverse=False):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(patches)
        return scaler.transform(patches), scaler

    if inverse:
        return scaler.inverse_transform(patches)
    else:
        return scaler.transform(patches)


from sklearn.neighbors import NearestNeighbors


"""
    returns index of nearest neighbour for some vector
"""
def apply_nearest_neighbor(patches, nbrs=None):
    if nbrs is None:
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(patches)
        return nbrs

    return nbrs.kneighbors(patches, return_distance=False).T[0]


"""
    # setup workflow:
    # =========
    #     apply scaling to img (gaussian_pyramid)
    #     for each scale:
    #         for each patch size:
    #             get patches
    #             apply standtard_scaler
    #             apply pca
    #             apply nearest_neighbor
        
        
    when searching:
    ---------------
        get patch you want to search for
        apply standard_scaler
        apply pca
        apply nearest neighbor to get neibor
        apply inverse pca
        apply inverse standard scaler
        
        voila, you have the patch xD
        
"""