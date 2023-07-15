#Using Tensorflow Version 1.x
#Code to be run with Colab or Jupyter Notebook.

%tensorflow_version 1.x
import tensorflow
import pylab as plt
import numpy as np
import nibabel as nib
%matplotlib inline 
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 16, 8
import glob
import cv2 as cv


from deepbrain import Extractor
import SimpleITK as sitk
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.align import (affine_registration, center_of_mass, translation,
                        rigid, affine, register_dwi_to_template)
from nilearn import image as nli



files = glob.glob("/content/drive/MyDrive/Prediction of Autism /Data/IXI-T1/*")

def load_img(img_path):#Function takes in one raw image and outputs its fdata.
    return nib.load((img_path))


class Template:
    def __init__(self, template):
        self.template = template
        self.template_data = template.get_fdata()
        self.template_grid2world = template.affine
    
    def distplay_template(self):
        plt.imshow(self.template_data[95, :, :])

class Preprocessing:
    def __init__(self, template: Template, save_path: str, img, segment: bool = True, interactive: bool = False):
        self.img = img #Actual image
        self.img_data = img.get_fdata()
        self.img_header = img.header
        self.img_grid2world = img.affine

        self.segment = segment
        self.save_path = save_path
        
    def smoothing(self, fwhm = 1, plot = False): #1mm smoothing
        smooth_img = nli.smooth_img(self.img, fwhm)
        if plot:
            plotting.plot_epi(smoothed_img, title="Smoothing %imm" % fwhm, display_mode='z', cut_coords=[-20, -10, 0, 10, 20, 30], cmap='magma')
        return smooth_img #Returns NIFTI image.
    
    def brainExtraction(self, PROB = 0.5):
        self.img = self.smoothing()
        self.img_data = self.img.get_fdata()
        self.img_grid2world = self.img.affine 

        ext = Extractor()
        prob = ext.run(self.img_data)
        mask = prob > PROB # Probability of a voxel being a tissue.
        extracted_brain = self.img_data * mask
        return extracted_brain #Returns numpy array

    def affineRegistration(self, display_transformation = False):
        self.img_data = self.brainExtraction()

        pipeline = [center_of_mass, translation, rigid, affine]
        nbins = 32
        level_iters = [1000, 100, 10] # Gaussian pyramid.
        sigmas = [3.0, 1.0, 0.0] # Smoothing params 3mm, 1mm, original
        factors = [4, 2, 1] # x//4, x//2, x//1

        transformed_img, reg_affine = affine_registration(moving = self.img_data, static = template.template_data,
                                                  moving_affine = self.img_grid2world,
                                                  static_affine = template.template_grid2world,
                                                  nbins = nbins, metric = "MI",
                                                  pipeline = pipeline, level_iters = level_iters,
                                                  sigmas = sigmas, factors = factors)
        
        if display_transformation:

            regtools.overlay_slices(template_data, transformed_img, None, 0,
                            "template_data", "Moving")
            regtools.overlay_slices(template_data, transformed_img, None, 1,
                                    "template_data", "Moving")
            regtools.overlay_slices(template_data, transformed_img, None, 2,
                                    "template_data", "Moving")
            
        return (transformed_img, reg_affine)


    def scaling(self, save: bool = True): #Min-max scaling
        transformed_img, _ = self.affineRegistration()
        rescaled = ((transformed_img - transformed_img.min()) * 255. / (transformed_img.max() - transformed_img.min())).astype(np.uint8)
        rescaled_img = nib.nifti1.Nifti1Image(rescaled, affine = self.img_grid2world, header = self.img_header) #Header change??
        
        if save:
            nib.save(rescaled_img, self.save_path + "rescaled_img.nii.gz")

        return rescaled_img
    
    def biasFieldCorrection(self):
        self.scaling()
        img = sitk.ReadImage(self.save_path + "rescaled_img.nii.gz") #If saved in a file/folder.
        mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
        sitk.WriteImage(mask_img, self.save_path + "mask_img.nii.gz")
        img = sitk.Cast(img, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(img, mask_img)
        sitk.WriteImage(corrected, self.save_path + "bias_corrected_img.nii.gz")

    def to_unit8(self, data):
        data = data.astype(np.float)
        data[data < 0] = 0
        return ((data - data.min()) * 255.0 / data.max()).astype(np.uint8)    
    
    def histogramEqualization(self): 
        self.biasFieldCorrection()
        hist_eq_img = self.to_unit8(nib.load("/content/drive/MyDrive/Prediction of Autism /Data/Saved Objects/bias_corrected_img.nii.gz").get_fdata())
        for slice_idx in range(hist_eq_img.shape[2]):
            hist_eq_img[:, :, slice_idx] = cv.equalizeHist(hist_eq_img[:, :, slice_idx])
        return hist_eq_img    

    def intensity_mrf_segmentation(self, nclass = 3, beta = 0.1):
      hmrf = TissueClassifierHMRF()
      hist_eq_img = self.histogramEqualization()
      initial_segmentation, final_segmentation, PVE = hmrf.classify(hist_eq_img, nclass, beta)
      return final_segmentation
        
    def process(self):
      return self.intensity_mrf_segmentation()

if __name__ == "__main__":
    
    template = load_img("/content/drive/MyDrive/Prediction of Autism /Data/Template/template.nii.gz")
    template = Template(template = template)

    for idx in range(15, 29): #Kanishk; Change the file index accordingly
      img = load_img(files[idx]) #Iterate over all images one by one; use for loop
      preprocess = Preprocessing(template = template, save_path= "/content/drive/MyDrive/Prediction of Autism /Data/Saved Objects/", img = img)
      with tf.device('/device:GPU:0'):
        final_img = preprocess.process()
        np.save("/content/drive/MyDrive/Prediction of Autism /Data/Preprocessed Files" + idx, final_img)
