import cv2
import numpy as np
import os
import sys
from functools import partial
import pytesseract
import textdistance as td
import pytesseract
import argparse
from tqdm import tqdm
import glob
from math import exp



# BEGIN ------ TEXT RETRIEVAL

def clean_text(text:str) -> str:
    if text is None:
        return None
    text = "".join(text.split()).replace(":", "").replace("-", "").replace(",", "").replace(".", "").replace("|", "")
    text = text.replace(":", "").replace("-", "").replace(",", "").replace(".", "").replace("|", "")
    return text.lower()
    

def extract_text_tesseract(image:np.ndarray) -> np.ndarray:
    """
    Extract 3D histogram from BGR image color space
    
    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8
        bins: number of bins to use for histogram
        mask: check _descriptor(first function in file)
        
    Returns: 
        3D histogram features flattened into a 
        1D array of type np.float32
    """

    # PREPROCESSING
    # Upsampling
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # To grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoising
    image = cv2.GaussianBlur(image , (3, 3), 0) 
    # Binarization
    (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY ) 

    #Tesseract Prediction
    predicted_text = pytesseract.image_to_string(image, config = '--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    predicted_text = clean_text(predicted_text)

    #Remove Digits
    remove_digits = predicted_text.maketrans('01258', 'OiZSB')#('', '', '0123456789')
    predicted_text = predicted_text.translate(remove_digits)
    remove_digits = predicted_text.maketrans('', '', '0123456789')
    predicted_text = predicted_text.translate(remove_digits)
    
    return predicted_text

# END ------ TEXT RETRIEVAL




TEXT_EXTRACTORS = {
    "tesseract": extract_text_tesseract,
    }

TEXT_SIMILARITIES = {
    "ratcliff_obershelp": td.ratcliff_obershelp,
    "levenshtein": td.levenshtein.normalized_similarity,
    "cosine": td.cosine,
    }


def extract_text(image:np.ndarray, extractor:str) -> str:
    """
    Extracts text from image using the method specified

    Args:
        image: (H x W x C) 3D BGR image array of type np.uint8 from which extract the text
        extractor: method used to compute features

    Returns: 
        str:   text extracted from the image
    """
    return TEXT_EXTRACTORS[extractor](image)

def compare_texts(text1:str, text2:str, similarity:str) -> float:
    """
    Extracts text from image using the method specified

    Args:
        text1: text to compare
        text2: text to compare

    Returns: 
        float similarity value
    """
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    if text1 is None or text2 is None:
        return 1.0
    sim = TEXT_SIMILARITIES[similarity](text1, text2)
    #1/(1+e^(-50(x-0.05)))
    sim = 1 / (1 + exp(-50*(sim-0.05)))
    return 1 - sim




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text extraction test')

    parser.add_argument(
        "--dataset", type=str,
        help = "Dataset where cropped textboxes are in jpg format together with a txt file with the GT text.")

    parser.add_argument(
        "--extractor", help = "Extractor used to extract text from the images.", choices=list(TEXT_EXTRACTORS.keys()))

    parser.add_argument(
        "--comparer", help = "Similarity metric used to compare extracted text and GT.", choices=list(TEXT_SIMILARITIES.keys()))
    
    #parser.add_argument(
    #    "--find_textbox", "c", action="store_true",
    #    help = "Similarity metric used to compare extracted text and GT.", choices=list(TEXT_SIMILARITIES.keys()))

    args = parser.parse_args()
    print(args)
    
    sim_sum, i = 0, 0
    for textfile_path in tqdm(sorted(glob.glob(os.path.join(args.dataset, "*.txt"), recursive=False))):
        # Content of .txt file
        with open(textfile_path, "r") as f:
            GT = f.read()
                          
        # Content of .txt file without spaces
        #GT = "".join(GT.split()).replace(":", "").replace("-", "").replace(",", "").replace(".", "").replace("|", "").replace(" ", "")

        # Text extraction
        image_path = textfile_path.replace("txt", "jpg")
        textbox_img = cv2.imread(image_path)
        
        text = extract_text(textbox_img, args.extractor)

        sim = compare_texts(GT, text, args.comparer)
        if sim < 0.85:
            print(f"{GT} -> {text} ({sim})")
        sim_sum += sim
        i += 1
        
    print(sim_sum/i)
        
        
        
        
        
        