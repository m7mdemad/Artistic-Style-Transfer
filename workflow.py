import numpy as np
from commonfunctions import *
from hist_match import color_transform
from skimage.transform import pyramid_gaussian
from skimage.util import noise
from NN import *
from skimage.transform import rescale
import pandas as pd
from irlsV2 import IRLS
from animated_plot import show_animated
from segmentation import get_segmentation
#from Domain_Transfer_2 import *
import cv2

patch_sizes = np.array([33, 21, 13, 9])
subsampling_gaps = np.array([28, 18, 8, 5])
r = 0.8  # robust fusion for IRLS
I_alg = 2

"""
    *takes conent, style and segmentation images and returns gaussian pyramid built from them
        with maximum depth of L_max
    * It applies color transfer to the content image before building its pyramid
    * It initializes the estimate image with the noise and scaling required
"""


def initialize(content, style, segmentation,L_max):
    # apply color transfer from S to C
    new_content = color_transform(content, style)
    # Build the Gaussian pyramids of C, S, and W
    images = pd.DataFrame(index=["L" + str(i) for i in range(L_max + 1)], columns=['content', 'style', 'segmentation', 'estimation'])
    images['original'] = list(pyramid_gaussian(content, multichannel=True, max_layer=L_max))
    images['content'] = list(pyramid_gaussian(new_content, multichannel=True, max_layer=L_max))
    images['style'] = list(pyramid_gaussian(style, multichannel=True, max_layer=L_max))
    images['segmentation'] = list(pyramid_gaussian(segmentation, multichannel=False, max_layer=L_max))
    #mages['segmentation'] = [None]*(L_max+1)

    # initialize X with additive gaussian to content ∼ N (0, 50)
    estimated_image = noise.random_noise(new_content, mode="gaussian", mean=0, var=50 / 256)  # normalized variance because image is normalized
    images['estimation'] = list(pyramid_gaussian(estimated_image, multichannel=True, max_layer=L_max))
#    images['estimation'][images.index[-1]] = rescale(estimated_image, 1 / (2 ** L_max), multichannel=True,anti_aliasing=True, mode='constant', cval=0)

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
            patches = get_patches_aux(image, patch_size=patch_sizes[index], subsampling_gap=subsampling_gaps[index])
        else:
            patches = get_patches(image, patch_size=patch_sizes[index], subsampling_gap=patch_sizes[index])
        return patches

def get_nn_style_patch_from_indices(style_patches, nn_indices, estimation_p_shape):
    s_p_shape       = style_patches.shape
    s_patches_count   = s_p_shape[0] * s_p_shape[1]
    nearest_patches = np.empty(estimation_p_shape)
    nearest_patch_index = 0
    for index in nn_indices:
        rotation= index // s_patches_count
        r       = index % s_patches_count
        v_index = r // s_p_shape[0]
        h_index = r % s_p_shape[1]
        nn_row = nearest_patch_index // estimation_p_shape[1]
        nn_col = nearest_patch_index % estimation_p_shape[1]
        nearest_patches[nn_row,nn_col] = np.rot90(style_patches[v_index,h_index], k=rotation, axes=(0,1))
        nearest_patch_index += 1
    return nearest_patches
        
        

"""
    * takes style pyramid and converts it into patches for each layer and each patch size
    * It trains the models for standard_scaler, pca and nn on those patches
    * It returns the trained models objects  and the pyramid of patches acquired
"""

def prepare_style_patches(style_pyramid,L_max):
    
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
    I = np.ones_like(segmentation.shape[0])
    w_I_ = 1/(segmentation + I)
#    return np.stack([((np.linalg.inv(segmentation[:,:,i] + identity_matrix)).dot(estimation[:,:,i] + segmentation[:,:,i].dot(content[:,:,i]))) for i in range(3) ], axis=(2))
#    return np.stack([((np.linalg.inv(segmentation[:,:,i] + identity_matrix)).dot(estimation[:,:,i] + segmentation[:,:,i]*content[:,:,i])) for i in range(3) ], axis=(2))
    return np.stack([(w_I_ * (estimation[:,:,i] + segmentation*content[:,:,i])) for i in range(3) ], axis=(2))


def style_transfer(content_path, style_path, I_irls, I_alg, seg_fac,L_max):
    content = io.imread(content_path).astype('float64')/255
    content = cv2.resize(content, (400,400))
    style = io.imread(style_path).astype('float64')/255
    style = cv2.resize(style, (400, 400))
    segmentation = get_segmentation(content_path)*seg_fac + 0.25*seg_fac
#    segmentation = np.zeros(content.shape[:-1])
    
    images = initialize(content, style, segmentation,L_max)
    scaler_objects, pca_objects, nn_objects, style_patches = prepare_style_patches(images['style'],L_max=L_max)
    for layer in reversed(style_patches.index):
        for index, patch_size in enumerate(style_patches.columns):
            # add another loop for IRLS itarations
            scaler  = scaler_objects[patch_size][layer]
            pca     = pca_objects[patch_size][layer]
            nn      = nn_objects[patch_size][layer]
            current_style_patches = style_patches[patch_size][layer]
            
            for _ in range(I_alg):
                # convert estimation image of the current layer to patches we can operate on
                estimation_image = noise.random_noise(images['estimation'][layer], mode="gaussian", mean=0, var=100 / 256)  # normalized variance because image is normalized

                current_estimation_patches  = image_to_patches(estimation_image, is_pyramid=False, index=index)
                flat_curr_estimation_patches= flatten_patches(current_estimation_patches, rotate=False)
                scaled_estimation_patches = apply_standard_Scaler(flat_curr_estimation_patches, scaler=scaler)
                reduced_estimation_patches = apply_pca(scaled_estimation_patches, pca=pca)
                nn_indices = apply_nearest_neighbor(reduced_estimation_patches, nbrs=nn)
                nn_patches = get_nn_style_patch_from_indices(current_style_patches, nn_indices, current_estimation_patches.shape)
                # IRLS
                estimation_image = IRLS(images['estimation'][layer], nn_patches, patch_size, subsampling_gaps[index], I_irls)
                e1 = np.copy(estimation_image)
                # content fusion
                estimation_image = content_fusion(images['content'][layer], estimation_image, images['segmentation'][layer])
                e2 = np.copy(estimation_image)
                # color transfer
                estimation_image = color_transform(estimation_image , images['style'][layer])
                e3 = np.copy(estimation_image)
                # domain transform filter
#                sigma_s = 100
#                sigma_r = 3
#
#                estimation_image = Iterative_C(estimation_image * 255, sigma_s, sigma_r,3)/255
#                e4 = np.copy(estimation_image)
                
                show_animated([images['style'][layer], images['original'][layer], estimation_image], ["style", "content", "estimation"] )
                # print("style max", images['style'][layer].max(),"style min", images['style'][layer].min())
                # print("content max", images['content'][layer].max(),"content min", images['content'][layer].min())
                # print("est max", images['estimation'][layer].max(),"est min", images['estimation'][layer].min())
#                estimation_image = noise.random_noise(estimation_image, mode="gaussian", mean=0, var=50 / 256)  # normalized variance because image is normalized
#                e5 = estimation_image
#                show_animated([e1, e2, e3, e4, e5], ["irls", "content fusion", "color transfer", "domain transform", "noise"])
                images['estimation'][layer] = estimation_image

            # scaling up
        images['estimation'][list(images.index).index(layer) - 1] =\
                rescale(images['estimation'][layer], 2, multichannel=True,anti_aliasing=True, mode='constant', cval=0)
#    show_images([images['style'][layer], images['original'][layer], images['estimation']['L0']],["style", "content", "result"])
#    plt.pause(500)
    return images['estimation']['L0']
if __name__ == "__main__":
    main()
