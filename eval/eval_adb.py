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

experiment_path = '../nam-dev/insights/outputs/NEW_ATTACK/'
ids = list(os.listdir(experiment_path))

prompts = ['a_photo_of_sks_person', 'a_dslr_portrait_of_sks_person']

accumulators = [Accumulator(), Accumulator(), Accumulator(), Accumulator(), Accumulator(), Accumulator()]
# fdfr_accumulator = Accumulator()
# brisque_accumulator = Accumulator()
# psnr_accumulator = Accumulator()
# lpips_accumulator = Accumulator()

for prompt in prompts:
    for accumulator in accumulators:
        accumulator.reset()

    for id in ids:
        generated_target_folder = f'../adb/outputs/ASPL_PNG_THESIS/{id}/dreambooth/checkpoint-1000/images/{prompt}'
        generated_reference_folder = f'../adb/db_dataset/{id}/set_A'

        protected_target_folder = f'../adb/outputs/ASPL_PNG_THESIS/{id}/adversarial/noise-ckpt/50'
        protected_reference_folder = f'../adb/outputs/ASPL_PNG_THESIS/{id}/adversarial/image_before_adding_noise'

        try:
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
        except:
            continue
            
        print(f"FDFR: {fdfr_score}")
        print(f"ISM: {ism_score}")
        print(f"BRISQUE: {brisque_score}")
        print(f"PSNR: {psnr_score}")
        print(f"LPIPS: {lpips_score}")
        print(f"SSIM: {ssim_score}")

        result = {
            'id': id,
            'prompt': prompt,
            'fdfr': fdfr_score,
            'ism': ism_score,
            'brisque': brisque_score,
            'lpips': lpips_score,
            'psnr': psnr_score,
            'ssim': ssim_score
        }
        
        scores = [fdfr_score, ism_score, brisque_score, psnr_score, lpips_score, ssim_score]
        for i in range(len(scores)):
            accumulators[i].accumulate(scores[i])

    with open('./log_adb.txt', 'a') as log:
        log.write(f'prompt = \'{prompt}\'\n')
        for accumulator in accumulators:
            log.write(f'{accumulator.average()}\n')

        log.write('\n')