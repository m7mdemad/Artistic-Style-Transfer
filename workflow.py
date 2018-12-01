import numpy as np
from commonfunctions import *
from hist_match import color_transform
from skimage.transform import pyramid_gaussian
from skimage.util import noise
from NN import *
from skimage.transform import rescale



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
    content_pyramid = list(pyramid_gaussian(new_content, multichannel=True, max_layer=L_max))
    style_pyramid = list(pyramid_gaussian(style, multichannel=True, max_layer=L_max))
    # segmentation_pyramid = list(pyramid_gaussian(segmentation, multichannel=True, max_layer=L_max))
    segmentation_pyramid = [None]*(L_max+1)

    # initialize X with additive gaussian to content ∼ N (0, 50)
    estimated_image = noise.random_noise(content, mode="gaussian", mean=0,
                                         var=50 / 256)  # normalized variance because image is normalized
    # estimated_image_pyramid = list(pyramid_gaussian(estimated_image, multichannel=True, max_layer=L_max))
    estimated_image = rescale(estimated_image, 1 / (2 ** L_max), multichannel=True,anti_aliasing=True, mode='constant', cval=0)

    return content_pyramid, style_pyramid, segmentation_pyramid, estimated_image


"""
converts images pyramid into a pyramid of their constructing patches

overlapping_patches: a flag to whether we want overlapping patches from image or not
"""


def image_to_patches(image, overlapping_patches=True, is_pyramid=True):
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
            patches = get_patches(image, patch_sizes=patch_sizes, subsampling_gaps=subsampling_gaps)
        else:
            patches = get_patches(image, patch_sizes=patch_sizes, subsampling_gaps=patch_sizes)
        return patches

"""
    * takes style pyramid and converts it into patches for each layer and each patch size
    * It trains the models for standard_scaler, pca and nn on those patches
    * It returns the trained models objects  and the pyramid of patches acquired
"""

def prepare_style_patches(style_pyramid):
    patches_pyramid = image_to_patches(style_pyramid)

    scaler = []
    pca = []
    nbrs = []
    for layer_index, layer in enumerate(patches_pyramid):
        layer_scaler_objs = []
        layer_pca_objs = []
        layer_nbrs_objs = []
        for patches_index, patches in enumerate(layer):
            patches_pyramid[layer_index][patches_index], scaler_obj = apply_standard_Scaler(patches)
            layer_scaler_objs.append(scaler_obj)
            patches_pyramid[layer_index][patches_index], pca_obj = apply_pca(patches)
            layer_pca_objs.append(pca_obj)
            nbrs_obj = apply_nearest_neighbor(patches)
            layer_nbrs_objs.append(nbrs_obj)
        scaler.append(layer_scaler_objs)
        pca.append(layer_pca_objs)
        nbrs.append(layer_nbrs_objs)

    return scaler, pca, nbrs, patches_pyramid


def main():
    
    content = io.imread(r"images/house 2-small.jpg")
    style = io.imread(r"images/starry-night - small.jpg")

    # TODO: Add segmentation here
    segmentation = None
    # show_images([content, style, segmentation])
    
    content_pyramid, style_pyramid, segmentation_pyramid, estimation_image = initialize(content, style, segmentation)
    scaler_objects, pca_objects, nn_objects, style_patches_pyramid = prepare_style_patches(style_pyramid)
    estimation_patches_pyramid = image_to_patches(estimation_image, is_pyramid=False)

    for layer_index, content_layer, style_layer, segmentation_layer, estimation_patches_layer, \
        scaler_objects_layer, pca_objects_layer, nn_objects_layer, style_patches_layer in reversed(
        list(enumerate(
            zip(content_pyramid, style_pyramid, segmentation_pyramid, estimation_patches_pyramid,
                scaler_objects, pca_objects, nn_objects, style_patches_pyramid)))):
        for patch_size_index, patch_size, gap, scaler, pca, nn, style_patches, estimation_patches in \
                enumerate(zip(patch_sizes, subsampling_gaps, scaler_objects_layer, pca_objects_layer,
                              nn_objects_layer, style_patches_layer, estimation_patches_layer)):

            nn_style = np.zeros_like(estimation_image)
            scaled_estimation_patches = apply_standard_Scaler(estimation_patches_layer, scaler=scaler)
            reduced_estimation_patches = apply_pca(scaled_estimation_patches, pca=pca)
            nn_indices = apply_nearest_neighbor(reduced_estimation_patches, nbrs=nn)
            nn_reduced_style_patches = style_patches[nn_indices]
            nn_unreduced_estimation_patches = apply_pca(nn_reduced_style_patches, pca=pca, inverse=True)
            nn_style_patches = apply_standard_Scaler(nn_unreduced_estimation_patches, scaler=scaler, inverse=True)

            vertical_patch_count = nn_style.shape[0] // patch_size
            horizontal_patch_count = nn_style.shape[1] // patch_size
            patch_index = 0
            for v_index in range(vertical_patch_count):
                for h_index in range(horizontal_patch_count):
                    nn_style[v_index * patch_size: v_index * patch_size + patch_size,
                    h_index * patch_size: h_index * patch_size + patch_size] = \
                        nn_style_patches[patch_index].reshape((patch_sizes, patch_sizes, 3))
                    patch_index += 1

            # nn_style image should be ready by now


if __name__ == "__main__":
    main()
