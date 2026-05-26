import os
import sys

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.eval.metrics import calculate_style_accuracy, calculate_melody_distance

if __name__ == '__main__':

    ORIGINALS_DIR = "path/to/original/bach/files"
    TRANSFERRED_DIR = "path/to/transferred/chopin/style/files"
    
    CLASSIFIER_CKPT_PATH = "checkpoints/style-classifier-best.ckpt"
    TARGET_STYLE_LABEL = 1 

    transferred_files = [os.path.join(TRANSFERRED_DIR, f) for f in os.listdir(TRANSFERRED_DIR)]
    style_acc = calculate_style_accuracy(transferred_files, TARGET_STYLE_LABEL, CLASSIFIER_CKPT_PATH)
    print(f"\nStyle Match Accuracy: {style_acc:.2f}%\n")

    print("Calculating Melody Preservation Similarity")
    total_similarity = 0
    num_files = 0
    for filename in os.listdir(TRANSFERRED_DIR):
        original_path = os.path.join(ORIGINALS_DIR, filename)
        transferred_path = os.path.join(TRANSFERRED_DIR, filename)
        
        if os.path.exists(original_path):
            similarity = calculate_melody_distance(original_path, transferred_path)
            print(f"'{filename}': Melody Similarity = {similarity:.3f}")
            total_similarity += similarity
            num_files += 1
    
    avg_similarity = (total_similarity / num_files) if num_files > 0 else 0
    print(f"\nAverage Melody Similarity: {avg_similarity:.3f}")