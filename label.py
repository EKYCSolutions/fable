import os
import warnings
import argparse
from glob import glob
from concurrent.futures import ThreadPoolExecutor
#
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import face_recognition
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
#
from utils import (
    read_config,
    gather_samples,
    image_to_base64,
    create_output_model,
    construct_system_message,
)
from tracker import Tracker
#
parser = argparse.ArgumentParser(description='Start the labeling process.')
parser.add_argument('data_dir', type=str, help='Path to data directory')
#
parser.add_argument('-c', '--config', type=str,
                    help='Path to config file.', default="config.yaml")
parser.add_argument('-o', '--output_dir', type=str,
                    help='Path to the output directory.', default="out")
parser.add_argument('-w', '--workers', type=int,
                    help='The number of workers for processing the samples', default=4)
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Wether to print out info')
args = parser.parse_args()
#
os.makedirs(args.output_dir, exist_ok=True)

output_filepath = os.path.join(args.output_dir, "annotation.csv")
config = read_config(args.config)
tracker = Tracker(
    output_dir=args.output_dir,
    batch_size=args.workers,
)
llm = ChatOllama(model=config['configurations']['model'])
OutputModel = create_output_model(config['accessories'])
SYSTEM_MESSAGE = construct_system_message(config['accessories'])
#

def process_sample(sample_filepath: str):
    image = Image.open(sample_filepath)
    if config['configurations']['detect_faces']:
        bboxes = face_recognition.face_locations(image)
    else:
        bboxes = [(0, image.width, image.height, 0)]  # the whole image
    #
    image = np.array(image)

    annotations = []
    for person_id, bbox in enumerate(bboxes):
        ymin, xmax, ymax, xmin = bbox
        face_image = image[ymin:ymax, xmin:xmax]
        face_image = Image.fromarray(face_image)

        image_base64 = image_to_base64(face_image)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            ("human", [
                {"type": "text", "text": "Describe the following image:"},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ])
        ])
        chain = prompt | llm.with_structured_output(OutputModel)
        response = chain.invoke({})
        annotations.append({
            "filename": sample_filepath,
            "person_id": person_id,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            **response.model_dump(),
        })

    pd.DataFrame(annotations).to_csv(
        output_filepath,
        mode="a",
        header=not os.path.exists(output_filepath),
        index=False
    )


if __name__ == "__main__":
    all_samples = gather_samples(
        data_dir=args.data_dir,
        extensions=config['configurations']['image_extensions']
    )
    tracker.add_samples(all_samples)

    num_total = len(all_samples)
    num_pending = tracker.pending_count()
    
    with tqdm(total=num_total, initial=(num_total-num_pending), desc="Processing items") as pbar:
        while tracker.pending_count() > 0:
            sample_paths = tracker.get_batch()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(process_sample, sample_paths))

            for sample in sample_paths:
                tracker.mark_done(sample)
                pbar.update(1)