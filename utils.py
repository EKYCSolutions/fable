import os
import base64
from glob import glob
from io import BytesIO
from typing import List, Tuple, Dict, Any
from pydantic import create_model, BaseModel, Field
#
import yaml
from PIL import Image

def read_config(filepath: str) -> Dict[str, Any]:
    """
    Reads a configuration file given by `filepath`.

    :param filepath: Path to the yaml config file.
    :returns: A dictionary representing the content of the config file.
    """
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
        return config
    
def gather_samples(data_dir: str, extensions: List[str]) -> List[str]:
    """
    Recursively find all samples within `data_dir` with all the extensions within `extensions`.

    :param data_dir: Base path
    :param extensions: List of extensions to gather.
    :returns: A list of file paths of all the match samples.
    """
    #
    all_samples = []
    for extension in extensions:
        all_samples += list(glob(os.path.join(data_dir, "**", f"*.{extension}"), recursive=True))
    #
    return all_samples

def image_to_base64(image: Image.Image) -> str:
    """
    Converts PIL image to base64 string.

    :param image: The source image.
    :returns: The base64 representation of `image`
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64\
        .b64encode(buffered.getvalue())\
        .decode("utf-8")

def create_output_model(accessories: Dict[str, str]) -> BaseModel:
    """
    Creates a dynamic Pydantic BaseModel given the list of `accessories`.

    :param accessories: The accessories dictionary whose key is the accessory name and value is the description of the accessory.
    :returns: A BaseModel
    """
    fields = {
        name: (int, Field(0, description=desc))
        for name, desc in accessories.items()
    }
    return create_model("OutputModel", **fields)

def construct_system_message(accessories: Dict[str, str]) -> str:
    message = """
You are FABLE, a vision-language model for face accessory labeling.
Given an image of a person face, determine if the following accessories are present:
"""
    #
    for i, (name, description) in enumerate(accessories.items()):
        message += "\n"
        message += f"{i+1}. {name}: {description}."
    #
    return message