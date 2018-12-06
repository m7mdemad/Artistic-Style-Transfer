import numpy as np
from commonfunctions import *
from hist_match import color_transform
from skimage.transform import pyramid_gaussian
from skimage.util import noise
from NN import *
from skimage.transform import rescale
import pandas as pd


L_max = 3
patch_sizes = np.array([33, 21, 13, 9])
subsampling_gaps = np.array([28, 18, 8, 5])
r = 0.8  # robust fusion for IRLS
I_alg = 3

"""
    *takes conent, style and segmentation images and returns gaussian pyramid built from them
        with maximum depth of L_max
    * It applies color transfer to the content image before building its pyramid
    * It initializes the estimate image with the noise and scaling required
"""


def initialize(content, style, segmentation):
    # apply color transfer from S to C
    new_content = color_transform(content, style)
    # Build the Gaussian pyramids of C, S, and W
    images = pd.DataFrame(index=["L" + str(i) for i in range(L_max + 1)], columns=['content', 'style', 'segmentation', 'estimation'])
    images['content'] = list(pyramid_gaussian(new_content, multichannel=True, max_layer=L_max))
    images['style'] = list(pyramid_gaussian(style, multichannel=True, max_layer=L_max))
    # iamges['segmentation'] = list(pyramid_gaussian(segmentation, multichannel=True, max_layer=L_max))
    images['segmentation'] = [None]*(L_max+1)

    # initialize X with additive gaussian to content ∼ N (0, 50)
    estimated_image = noise.random_noise(content, mode="gaussian", mean=0,
                                         var=50 / 256)  # normalized variance because image is normalized
    images['estimation'][images.index[-1]] = rescale(estimated_image, 1 / (2 ** L_max), multichannel=True,anti_aliasing=True, mode='constant', cval=0)

    return images


"""
converts images pyramid into a pyramid of their constructing patches

overlapping_patches: a flag to whether we want overlapping patches from image or not
"""


def image_to_patches(image, overlapping_patches=True, is_pyramid=True, index = 0):
    if is_pyramid:
        patches_pyramid = []
        for img in image:
            if overlapping_patches is True:
                patches = get_patches(img, patch_sizes=patch_sizes, subsampling_gaps=subsampling_gaps)
            else:
                patches = get_patches(img, patch_sizes=patch_sizes, subsampling_gaps=patch_sizes)
            patches_pyramid.append(patches)
        return patches_pyramid
    else:
        if overlapping_patches is True:
            patches = get_patches_aux(image, patch_sizes=patch_sizes[index], subsampling_gaps=subsampling_gaps[index])
        else:
            patches = get_patches(image, patch_sizes=patch_sizes[index], subsampling_gaps=patch_sizes[index])
        return patches

def get_nn_style_patch_from_indices(style_patches, nn_indices, estimation_p_shape):
    s_p_shape       = style_patches.shape
    s_patches_count   = s_p_shape[0] * s_p_shape[1]
    nearest_patches = np.empty(estimation_p_shape)
    nearest_patch_index = 0
    for index in nn_indices:
        rotation= index // s_patches_count
        r       = index % s_patches_count
        v_index = r % s_p_shape[1]
        h_index = r % s_p_shape[0]
        nn_row = nearest_patch_index % estimation_p_shape[1]
        nn_col = nearest_patch_index // estimation_p_shape[1]
        nearest_patches[nn_row,nn_col] = np.rot90(style_patches[v_index,h_index], k=rotation, axes=(0,1))
        nearest_patch_index += 1
    return nearest_patches
        
        

"""
    * takes style pyramid and converts it into patches for each layer and each patch size
    * It trains the models for standard_scaler, pca and nn on those patches
    * It returns the trained models objects  and the pyramid of patches acquired
"""

def prepare_style_patches(style_pyramid):
    
    style_patches = pd.DataFrame(data=image_to_patches(style_pyramid),
                                 index=["L" + str(i) for i in range(L_max + 1)], columns=patch_sizes)
    scaler  = pd.DataFrame(index=["L" + str(i) for i in range(L_max + 1)], columns=patch_sizes)
    pca     = pd.DataFrame(index=["L" + str(i) for i in range(L_max + 1)], columns=patch_sizes)
    nbrs    = pd.DataFrame(index=["L" + str(i) for i in range(L_max + 1)], columns=patch_sizes)
    for layer in style_patches.index:
        for patch_size in style_patches.columns:
            flat_patches = flatten_patches(style_patches[patch_size][layer])
            flat_patches , scaler_obj = apply_standard_Scaler(flat_patches)
            scaler[patch_size][layer] = scaler_obj
            flat_patches, pca_obj = apply_pca(flat_patches)
            pca[patch_size][layer] = pca_obj
            nbrs[patch_size][layer] = apply_nearest_neighbor(flat_patches)

    return scaler, pca, nbrs, style_patches

def content_fusion(content, estimation, segmentation):
    identity_matrix = np.identity(segmentation.shape[0])
    return np.stack([((np.linalg.inv(segmentation[:,:,i] + identity_matrix))
                      .dot(estimation[:,:,i] + segmentation[:,:,i].dot(c[:,:,i]))) for i in range(3) ], axis=(2))
    
    

def main():
    
    content = io.imread(r"images/house 2-small.jpg")
    style = io.imread(r"images/starry-night - small.jpg")

    # TODO: Add segmentation here
    segmentation = np.zeros_like(content)
    # show_images([content, style, segmentation])
    
    
    images = initialize(content, style, segmentation)
    scaler_objects, pca_objects, nn_objects, style_patches = prepare_style_patches(images['style'])
    for layer in reversed(style_patches.index):
        for index, patch_size in enumerate(style_patches.columns):
            # add another loop for IRLS itarations
            scaler  = scaler_objects[patch_size][layer]
            pca     = pca_objects[patch_size][layer]
            nn      = nn_objects[patch_size][layer]
            current_style_patches       = style_patches[patch_size][layer]
            for _ in range(I_alg):
                # convert estimation image of the current layer to patches we can operate on
                current_estimation_patches  = image_to_patches(images['estimation'][layer], is_pyramid=False)
                flat_curr_estimation_patches= flatten_patches(current_estimation_patches, rotate=False)
                scaled_estimation_patches = apply_standard_Scaler(flat_curr_estimation_patches, scaler=scaler)
                reduced_estimation_patches = apply_pca(scaled_estimation_patches, pca=pca)
                nn_indices = apply_nearest_neighbor(reduced_estimation_patches, nbrs=nn)
                nn_patches = get_nn_style_patch_from_indices(current_style_patches, nn_indices, current_estimation_patches.shape)
    
    
                # nn_style image should be ready by now
                # content fusion
                estimation_image = content_fusion(content, estimation, segmentation)
                # color transfer
                estimation_image = color_transform(estimation_image, images['style'][layer])
                # domain transform filter
                
            # scaling up
            images['estimation'][list(images.index).index(layer) - 1] =\
                rescale(estimated_image, 2, multichannel=True,anti_aliasing=True, mode='constant', cval=0)

if __name__ == "__main__":
    main()
