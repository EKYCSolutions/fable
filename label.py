import os
import argparse
from glob import glob
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
    image_to_base64,
    create_output_model,
    construct_system_message,
)
#
parser = argparse.ArgumentParser(description='Start the labeling process.')
parser.add_argument('data_dir', type=str, help='Path to data directory')
#
parser.add_argument('-c', '--config', type=str,
                    help='Path to config file.', default="config.yaml")
parser.add_argument('-o', '--output_filepath', type=str,
                    help='Path to the output csv file.', default="out.csv")
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Wether to print out info')
args = parser.parse_args()

config = read_config(args.config)


def get_samples():
    samples = list(glob(os.path.join(args.data_dir, "*")))
    for sample in tqdm(samples, total=len(samples), desc="Labeling"):
        image = Image.open(sample)
        #
        if config['configurations']['detect_faces']:
            bboxes = face_recognition.face_locations(image)
        else:
            bboxes = [(0, image.width, image.height, 0)]  # the whole image
        #
        image = np.array(image)
        for i, bbox in enumerate(bboxes):
            ymin, xmax, ymax, xmin = bbox
            face_image = image[ymin:ymax, xmin:xmax]
            face_image = Image.fromarray(face_image)

            yield (
                os.path.basename(sample),
                i,
                face_image,
                bbox,
            )


if __name__ == "__main__":
    #
    llm = ChatOllama(model=config['configurations']['model'])
    OutputModel = create_output_model(config['accessories'])
    SYSTEM_MESSAGE = construct_system_message(config['accessories'])

    data = []
    for filename, person_id, face_image, bbox in get_samples():
        ymin, xmax, ymax, xmin = bbox
        image_base64 = image_to_base64(face_image)
        #
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            ("human", [
                {"type": "text", "text": "Describe the following image:"},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ])
        ])
        chain = prompt | llm.with_structured_output(OutputModel)
        response = chain.invoke({})
        data.append({
            "filename": filename,
            "person_id": person_id,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            **response.model_dump(),
        })

    pd.DataFrame(data).to_csv(
        args.output_filepath,
        index=False,
    )
