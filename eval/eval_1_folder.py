import os
from itertools import product
from tqdm import tqdm

from metric import *

# PROTECTION QUALITY METRICS: FDFR, ISM, BRISQUE
#   Evaluate the quality of generated images from the DreamBooth model that is trained on protected images
#       - generated_target_folder: generated images of an identity
#       - generated_reference_folder: clean images of an identity

# PROTECTION QUALITY METRICS: SSIM, LPIPS, PSNR
#   Evaluate the quality of images that is protected by adversarial noise
#       - protected_target_folder: protected images
#       - protected_reference_folder: clean images (no protection)

generated_target_folder = f'../outputs/NEW_ATTACK_MEAN_DOT/5/eps=5e-2/self_latent+gaussian(std=2)_targeted/dreambooth/checkpoint-1000/images/a_photo_of_sks_person'
generated_reference_folder = f'../db_dataset/5/set_A'

protected_target_folder = f'../outputs/NEW_ATTACK_MEAN_DOT/5/eps=5e-2/self_latent+gaussian(std=2)_targeted/attacked_images'
protected_reference_folder = f'../db_dataset/5/set_B'

fdfr_score = FDFR.eval(
    generated_target_folder,
    log_info = True
)

ism_score = ISM.eval(
    generated_target_folder,
    generated_reference_folder,
    log_info = True
)

brisque_score = BRISQUE.eval(
    generated_target_folder
)

psnr_score = PSNR.eval(
    protected_target_folder,
    protected_reference_folder
)

lpips_score = LPIPS.eval(
    protected_target_folder,
    protected_reference_folder
)

ssim_score = SSIM.eval(
    protected_target_folder,
    protected_reference_folder
)

print(f"FDFR: {fdfr_score}")
print(f"ISM: {ism_score}")
print(f"BRISQUE: {brisque_score}")
print(f"PSNR: {psnr_score}")
print(f"LPIPS: {lpips_score}")
print(f"SSIM: {ssim_score}")