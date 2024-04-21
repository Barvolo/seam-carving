from copy import copy
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
import functools

    
def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        # Convert RGB to grayscale including the padded boundary
        grayscale_img = np.dot(np_img, self.gs_weights)
        
        return grayscale_img
        

    # @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        # Calculate the difference between each pixel and its next neighbor horizontally
        diff_x = self.resized_gs[:, :, 0] - np.roll(self.resized_gs[:, :, 0], shift=-1, axis=1)
        
        # Calculate the difference between each pixel and its next neighbor vertically
        diff_y = self.resized_gs[:, :, 0] - np.roll(self.resized_gs[:, :, 0], shift=-1, axis=0)
        
        # Compute the gradient magnitude as the square root of the sum of squares of the horizontal and vertical differences
        magnitude = np.sqrt(diff_x**2 + diff_y**2)

        # Ensure the gradient magnitude values are within the [0, 1] range
        magnitude_clipped = np.clip(magnitude, 0, 1)

        return magnitude_clipped
        

        
    def calc_M(self):
        pass
             
    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        self.gs = np.rot90(self.gs, -1 if clockwise else 1)
        self.rgb = np.rot90(self.rgb, -1 if clockwise else 1)
        self.resized_gs = np.rot90(self.resized_gs, -1 if clockwise else 1)
        self.resized_rgb = np.rot90(self.resized_rgb, -1 if clockwise else 1)
        self.cumm_mask = np.rot90(self.cumm_mask, -1 if clockwise else 1)
        self.seams_rgb = np.rot90(self.seams_rgb, -1 if clockwise else 1)
        self.idx_map_h = np.rot90(self.idx_map_h, -1 if clockwise else 1)
        self.idx_map_v = np.rot90(self.idx_map_v, 1 if clockwise else -1)
        self.E = np.rot90(self.E, -1 if clockwise else 1)

        self.h, self.w = self.w, self.h
        self.idx_map_h, self.idx_map_v = self.idx_map_v, self.idx_map_h
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()

    def init_mats(self):
        pass

    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            self.idx_map[i, s:] += 1

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        # Initialize M with the same shape as the initial energy map and set the first row of M
        M = np.zeros_like(self.E)
        M[0] = self.E[1]
        CL , CV, CR = self.calc_C_matrixes()
        # Iterate over each row (starting from the second)
        for i in range(1, M.shape[0]):
            M_up = M[i-1]
            M_left = np.roll(M_up, 1)
            M_right = np.roll(M_up, -1)
            # Calculate costs for each potential direction
            cost_left = M_left + CL[i]
            cost_up = M_up + CV[i]
            cost_right = M_right + CR[i]
            # Determine the minimum cost and its direction
            costs = np.stack([cost_left, cost_up, cost_right], axis=0)
            min_cost = np.min(costs, axis=0)
            # Update M and backtrack_mat
            M[i] = self.E[i] + min_cost
        
        return M

            
        
    # @NI_decor
    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        for _ in range(num_remove):
            self.init_mats()
            self.backtrack_seam()
            self.remove_seam()
            self.E = self.calc_gradient_magnitude()
            self.M = self.calc_M()
            self.seam_balance += 1


        



    def paint_seams(self):
        for s in self.seam_history:
            for i, s_i in enumerate(s):
                self.cumm_mask[self.idx_map_v[i,s_i], self.idx_map_h[i,s_i]] = False
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])

    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()
        #self.backtrack_mat = np.zeros_like(self.M, dtype=int)
        #self.mask = np.ones_like(self.M, dtype=bool)

    # @NI_decor
    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.rotate_mats(clockwise=True)
        self.seams_removal(num_remove)
        self.rotate_mats(clockwise=False)

    # @NI_decor
    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.seams_removal(num_remove)
        
    def calc_C_matrixes(self):
        left = np.roll(self.resized_gs[:, :, 0], -1, axis=1) 
        rigth = np.roll(self.resized_gs[:, :, 0], 1, axis=1)
        down = np.roll(self.resized_gs[:, :, 0], 1, axis=0)
       
        CV = np.abs(left - rigth)
        CL = CV + np.abs(rigth - down)
        CR = CV + np.abs(left - down)
        return CL, CV, CR
    
    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        j = np.argmin(self.M[self.h - 1])
        self.seam_history.append((self.h - 1, j))
        CL , CV, _ = self.calc_C_matrixes()
        
        for i in range(self.h - 1, 0, -1):
            if self.M[i, j] == self.E[i, j] + self.M[i - 1, j] + CV[i, j] : 
                j = j
            elif self.M[i, j] == self.E[i, j] + self.M[i - 1, j - 1] + CL[i, j] :
                if j != 0:  
                    j -= 1
            else:
                if j < self.w - 1:
                    j += 1
            self.seam_history.append((i - 1 , j))
            
  
    

        

    # @NI_decor
    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb) """
        # Create a mask for the seam removal
        new_width = self.w - 1

        # Update the cumulative mask
        rows, cols = zip(*[(x, y) for x, y in self.seam_history if y < self.w])

        # Initialize a 2D mask with True values
        mask_2d = np.ones((self.h, self.w), dtype=bool)
        mask_2d[rows, cols] = False

        # Apply the mask to grayscale and RGB images directly
        self.resized_gs = self.resized_gs[:, :, 0][mask_2d].reshape((self.h, new_width, 1))
        mask_3d = np.repeat(mask_2d[:, :, np.newaxis], 3, axis=2)
        self.resized_rgb = self.resized_rgb[mask_3d].reshape((self.h, new_width, 3))

        # Update index maps for each removed seam
        for row, col in self.seam_history:
            self.cumm_mask[self.idx_map_v[row, col], self.idx_map_h[row, col]] = False
            self.idx_map_h[row, col:-1] = self.idx_map_h[row, col + 1:]
            self.idx_map_v[row, col:-1] = self.idx_map_v[row, col + 1:]

        # Creating a 3D mask for visualizing seams on the original image
        if self.vis_seams:
            cumm_mask_rgb = np.stack([self.cumm_mask[:, :, 0]] * 3, axis=2)
            self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])
        
        self.w = new_width
        self.seam_history.clear()

        

    # @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")
    
    # @NI_decor
    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    # @NI_decor
    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    # @NI_decor
    @staticmethod
    # @jit(nopython=True)
    def calc_bt_mat(M, E, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.
        
        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        h, w = M.shape
        backtrack_mat[-1] = np.argmin(M[-1])
        for i in range(h-2, -1, -1):
            for j in range(w):
                backtrack_mat[i, j] = j + np.argmin(M[i+1, max(j-1, 0):min(j+2, w)]) - 1
        
        


class SCWithObjRemoval(VerticalSeamImage):
    def __init__(self, active_masks=['Gemma'], *args, **kwargs):
        import glob
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        self.active_masks = active_masks
        self.obj_masks = {basename(img_path)[:-4]: self.load_image(img_path, format='L') for img_path in glob.glob('images/obj_masks/*')}

        try:
            self.preprocess_masks()
        except KeyError:
            print("TODO (Bonus): Create and add Jurassic's mask")
        
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def preprocess_masks(self):
        """ Mask preprocessing.
            different from images, binary masks are not continous. We have to make sure that every pixel is either 0 or 1.

            Guidelines & hints:
                - for every active mask we need make it binary: {0,1}
        """
        for k in self.active_masks:
            self.obj_masks[k] = np.where(self.obj_masks[k] > 0.5, 1, 0)


        
    # @NI_decor
    def apply_mask(self):
        """ Applies all active masks on the image
            
            Guidelines & hints:
                - you need to apply the masks on other matrices!
                - think how to force seams to pass through a mask's object..
        """
        for obj in self.active_masks:
            for i in range(self.h):
                for j in range(self.w):
                    self.E[i, j] = 0 if self.obj_masks[obj][i, j] == 1 else 1
                    

    def init_mats(self):
        self.apply_mask() # -> added
        self.M = self.calc_M()
        self.h, self.w = self.resized_rgb.shape[:2]
        
        

    def reinit(self, active_masks):
        """ re-initiates instance
        """
        self.__init__(active_masks=active_masks, img_path=self.path)

    def remove_seam(self):
        """ A wrapper for super().remove_seam method. Takes care of the masks. 

        """
        # Create a mask for the seam removal
        new_width = self.w - 1

        # Update the cumulative mask
        rows, cols = zip(*[(x, y) for x, y in self.seam_history if y < self.w])

        # Initialize a 2D mask with True values
        mask_2d = np.ones((self.h, self.w), dtype=bool)
        mask_2d[rows, cols] = False

        # Apply the mask to grayscale and RGB images directly
        self.resized_gs = self.resized_gs[:, :, 0][mask_2d].reshape((self.h, new_width, 1))
        mask_3d = np.repeat(mask_2d[:, :, np.newaxis], 3, axis=2)
        self.resized_rgb = self.resized_rgb[mask_3d].reshape((self.h, new_width, 3))

        # Update index maps for each removed seam
        for row, col in self.seam_history:
            self.cumm_mask[self.idx_map_v[row, col], self.idx_map_h[row, col]] = False
            self.idx_map_h[row, col:-1] = self.idx_map_h[row, col + 1:]
            self.idx_map_v[row, col:-1] = self.idx_map_v[row, col + 1:]
            for obj in self.active_masks:
                    self.obj_masks[obj][row, col:-1] = self.obj_masks[obj][row, col+1:]

        # Creating a 3D mask for visualizing seams on the original image
        if self.vis_seams:
            cumm_mask_rgb = np.stack([self.cumm_mask[:, :, 0]] * 3, axis=2)
            self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])
        
        self.w = new_width
        self.seam_history.clear()

        

def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    return np.rint(np.multiply(orig_shape, scale_factors)).astype(int)

def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    seam_img_copy = copy.deepcopy(seam_img)
    seam_img_copy.seams_removal_vertical(shapes[0][1] - shapes[1][1])
    seam_img_copy.seams_removal_horizontal(shapes[0][0] - shapes[1][0])
    return seam_img_copy.resized_rgb

def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


