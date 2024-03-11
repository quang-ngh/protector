"""
    This file contains definition of evaluation metrics
"""

import os
import shutil
import numpy as np
import lpips as lps
import torch
import torchvision.transforms as transforms

from skimage.metrics import structural_similarity as compare_ssim
from deepface import DeepFace
from tqdm import tqdm
from brisque import BRISQUE as brs
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from utils import read_image

class Accumulator:
    """
        Accumulator helps with accumulating scalar values. Useful for calculating scores
    """
    def __init__(self):
        self.total = 0
        self.n = 0

    def accumulate(self, value):
        self.total += value
        self.n += 1
    
    def average(self):
        if self.n == 0:
            raise RuntimeError("<Accumulator.average()>: No elements")
        
        result = self.total / self.n

        return result
    
class Metric:
    def __init__(self):
        pass

class FDFR(Metric):
    @classmethod
    def eval(cls, target_folder, log_info = False, enable_progress_bar = True):
        target_files_name = os.listdir(target_folder)
        num_files = len(target_files_name)
        num_failure_cases = 0

        if num_files == 0:
            return 0
        
        for file_name in tqdm(target_files_name, desc = "FDFR.eval()", unit = "image"):
            target_file = f"{target_folder}/{file_name}"
            
            if os.path.isdir(target_file):
                continue

            try:
                DeepFace.extract_faces(
                    img_path = target_file, 
                    align = True
                )
            except:
                num_failure_cases += 1

        fdfr = num_failure_cases / num_files

        if log_info:
            print("FDFR.eval()")
            print(f"    target_folder = {target_folder}")
            print("----------")
            print(f"Num failure cases: {num_failure_cases}/{num_files}")
            print(f"FDFR Score: {fdfr:.2f}")
            print()

        return fdfr
        
class ISM(Metric):
    @classmethod
    def eval(cls, target_folder, reference_folder, log_info = False, enable_progress_bar = True):
        target_files_name = os.listdir(target_folder)
        num_target_files = len(target_files_name)
        target_face_vectors = []
        num_face_target_files = num_target_files
        no_face_target_files_name = []

        for file_name in target_files_name:
            target_file = os.path.join(target_folder, file_name)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = target_file,
                    model_name = 'ArcFace'
                )   
            except:
                num_face_target_files -= 1
                no_face_target_files_name.append(file_name)
            else:
                face_vector = face_embedding_info[0]['embedding']
                target_face_vectors.append(face_vector)
        
        reference_files_name = os.listdir(reference_folder)
        num_reference_files = len(reference_files_name)
        reference_face_vectors = []
        num_face_reference_files = num_reference_files
        no_face_reference_files_name = []

        for file_name in reference_files_name:
            reference_file = os.path.join(reference_folder, file_name)

            try:
                face_embedding_info = DeepFace.represent(
                    img_path = reference_file,
                    model_name = 'ArcFace'
                )   
            except:
                num_face_reference_files -= 1
                no_face_reference_files_name.append(file_name)
            else:
                face_vector = face_embedding_info[0]['embedding']
                reference_face_vectors.append(face_vector)
        
        def average_vector(lst):
            """
                Return the average vector (list) of all vectors inside lst

                Args:
                - lst: list[list]
            """
            if not lst:
                return []

            num_lists = len(lst)
            num_elements = len(lst[0])

            sums = [0] * num_elements

            for sub_list in lst:
                for i, element in enumerate(sub_list):
                    sums[i] += element

            average = [total / num_lists for total in sums]

            return average

        average_reference_face_vector = average_vector(reference_face_vectors)

        def cosine_similarity(vector1, vector2):
            """
                Return the cosine similarity between 2 vectors, scaled to [0, 1]
            """
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            similarity = dot_product / (norm_vector1 * norm_vector2)
            
            return (similarity + 1) / 2
        
        accumulator = Accumulator()

        progress_bar = target_face_vectors
        if enable_progress_bar:
            progress_bar = tqdm(target_face_vectors, desc = "Score calculation", unit = "image")

        for target_face_vector in progress_bar:
            score = cosine_similarity(target_face_vector, average_reference_face_vector)
            accumulator.accumulate(score)

        ism = accumulator.average()

        if log_info:
            print("ISM:")
            print(f"    target_folder = {target_folder}")
            print(f"    reference_folder = {reference_folder}")
            print(f"----------")
            print(f"Found {num_face_target_files} faces in {num_target_files} target_files")
            print(f"Found {num_face_reference_files} faces in {num_reference_files} reference_files")
            print("Files without faces: ")
            for file_name in no_face_target_files_name:
                print(f"- {file_name}")
            for file_name in no_face_reference_files_name:
                print(f"- {file_name}")
            
            print("----------")
            print(f"ISM Score: {ism}")
            print()

        return ism

class SER_FIQ(Metric):
    @classmethod
    def eval(cls, target_folder):
        return "Not implemented"

class BRISQUE(Metric):
    @classmethod
    def eval(cls, target_folder):
        accumulator = Accumulator()
        brisque_obj = brs(url = False)

        target_filenames = os.listdir(target_folder)
        
        for filename in tqdm(target_filenames, desc = "BRISQUE.eval() ", unit = " image"):
            image_path = f"{target_folder}/{filename}"

            if os.path.isdir(image_path):
                continue
            
            image = Image.open(image_path)
            image = np.array(image)

            score = brisque_obj.score(image)
            accumulator.accumulate(score)
        
        brisque = accumulator.average()

        return brisque

class IFR(Metric): # Identity and Fidelity Rating
    """
        Harmonic mean of ISM and (1 - scale(0, 1)(clamp(0, 100)(BRISQUE))). High when both ISM and (1 - BRISQUE) is high, indicate highly distorted image.
    """
    @classmethod
    def eval(cls, target_folder, reference_folder):
        ism = ISM.eval(
            target_folder,
            reference_folder,
            enable_progress_bar = False
        )

        brisque = BRISQUE.eval(target_folder)

        def clamp(n, min_value, max_value):
            return max(min_value, min(n, max_value))

        brisque = clamp(brisque, 0, 100) / 100
        
        ifr = 0 if ism == 0 or brisque == 0 else 2 * ism * (1 - brisque) / (ism + (1 - brisque))
            
        return ifr

class SSIM(Metric):
    @classmethod
    def eval(cls, target_folder, reference_folder):
        accumulator = Accumulator()

        target_images_name = sorted(os.listdir(target_folder))
        reference_images_name = sorted(os.listdir(reference_folder))

        for i in tqdm(range(len(target_images_name)), desc = "SSIM.eval() ", unit = " image"):
            target_image = Image.open(f"{target_folder}/{target_images_name[i]}")
            reference_image = Image.open(f"{reference_folder}/{reference_images_name[i]}")

            to_tensor = transforms.Compose(
                [transforms.PILToTensor()]
            )

            target_image = to_tensor(target_image).unsqueeze(0)
            reference_image = to_tensor(reference_image).unsqueeze(0)

            ssim = StructuralSimilarityIndexMeasure(data_range = 255)
            
            score = ssim(target_image, reference_image)

            accumulator.accumulate(score)

        avr_score = accumulator.average()

        return avr_score

class LPIPS(Metric):
    @classmethod
    def eval(cls, target_folder, reference_folder):
        accumulator = Accumulator()

        target_images_name = sorted(os.listdir(target_folder))
        reference_images_name = sorted(os.listdir(reference_folder))

        for i in range(len(target_images_name)):
            target_image = read_image(
                src = f"{target_folder}/{target_images_name[i]}",
                data_range = (-1, 1)
            )

            reference_image = read_image(
                src = f"{reference_folder}/{reference_images_name[i]}",
                data_range = (-1, 1)
            )

            target_image = target_image.cuda()
            reference_image = reference_image.cuda()

            lpips = LearnedPerceptualImagePatchSimilarity(net_type = "vgg")

            with torch.no_grad():
                score = lpips(target_image, reference_image)

            accumulator.accumulate(score)

        avr_score = accumulator.average()        

        return avg_score

class PSNR(Metric):
    @classmethod
    def eval(cls, target_folder, reference_folder):
        accumulator = Accumulator()

        target_images_name = sorted(os.listdir(target_folder))
        reference_images_name = sorted(os.listdir(reference_folder))

        for i in range(len(target_images_name)):
            target_img = read_image(f"{target_folder}/{target_images_name[i]}")
            ref_img = read_image(f"{reference_folder}/{reference_images_name[i]}")

            psnr = PeakSignalNoiseRatio()
            score = psnr(target_img, ref_img)

            accumulator.accumulate(score)

        avr_score = accumulator.average()

        return avr_score