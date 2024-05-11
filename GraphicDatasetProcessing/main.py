"""
This module processes and organizes images into a dataset based on user-defined parameters.
It supports resizing, optional grayscale conversion, and categorization of images.
"""
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass
import os
import argparse
from PIL import Image
import pandas as pd

@dataclass
class Config:
    """
    A class to store configuration parameters for image processing.

    Attributes:
        source_path: Path to the directory containing source images.
        output_path: Path where processed images and statistics will be saved.
        size: Target size for the output images as a tuple of width and height.
        grayscale: Whether to convert images to grayscale or not.
        image_format: The format in which to save the processed images ('png' or 'jpg').
    """
    source_path: str
    output_path: str
    size: Tuple[int, int]
    grayscale: bool
    image_format: str

class ImageProcessor:
    """
    A processor for images that handles resizing, optional grayscale conversion, and categorization
    into folders based on the original categories of images. It also generates statistics about the
    processed images.
    
    Attributes:
        source_path (str): The path where source images are located.
        output_path (str): The destination path for processed images.
        size (Tuple[int, int]): The dimensions to which images should be resized.
        grayscale (bool): Indicates if images should be converted to grayscale.
        image_format (str): The format for saving the processed images.
        category_stats (List[Dict[str, Union[int, str]]]): A list of dictionaries containing
            statistics about the processing of images in each category.
    """
    def __init__(self, config: Config) -> None:
        """
        Initializes an ImageProcessor instance using a configuration object.

        Args:
            config (Config): Configuration object containing source_path, output_path,
                             size, grayscale, and image_format.
        """
        self.source_path: str = config.source_path
        self.output_path: str = config.output_path
        self.size: Tuple[int, int] = config.size
        self.grayscale: bool = config.grayscale
        self.image_format: str = config.image_format
        self.category_stats: List[Dict[str, Union[int, str]]] = []

    def process_images(self) -> None:
        """
        Processes and organizes images into categories based on the configuration.
        """
        for category in os.listdir(self.source_path):
            category_path = os.path.join(self.source_path, category)
            if not os.path.isdir(category_path):
                continue

            output_category_path = os.path.join(self.output_path, category)
            os.makedirs(output_category_path, exist_ok=True)

            processed_count = 0
            abandoned_count = 0

            for filename in os.listdir(category_path):
                try:
                    input_image_path = os.path.join(category_path, filename)
                    output_image_path = os.path.join(
                        output_category_path,
                        f'{processed_count:03d}.{self.image_format}'
                    )

                    image = Image.open(input_image_path)
                    if self.grayscale:
                        image = image.convert('L')
                    else:
                        image = image.convert('RGB')
                    image = image.resize(self.size)

                    image.save(output_image_path)
                    processed_count += 1
                except IOError as e:
                    print(f"Failed to process {filename}: {e}")
                    abandoned_count += 1

            self.category_stats.append({
                'category': category,
                'img_count': processed_count,
                'abandoned_count': abandoned_count
            })

    def generate_csv(self) -> None:
        """
        Generates a CSV file with the statistics of processed images.
        """
        df = pd.DataFrame(self.category_stats)
        csv_path = os.path.join(self.output_path, 'image_statistics.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved statistics to {csv_path}')

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process and organize images into a dataset.')
    parser.add_argument('-s', '--source', type=str, \
        required=True, help='Source directory containing images.')
    parser.add_argument('-o', '--output', type=str, \
        required=True, help='Output directory for the processed images.')
    parser.add_argument('-sz', '--size', type=str, \
        required=True, help='Output image size in WIDTHxHEIGHT format.')
    parser.add_argument('-g', '--grayscale', \
        action='store_true', help='Convert images to grayscale.')
    parser.add_argument('-f', '--format', type=str, \
        choices=['png', 'jpg'], required=True, help='Output image format.')
    return parser.parse_args()

def main() -> None:
    """
    Main function to orchestrate the image processing and organization.
    """
    args = parse_arguments()
    size = tuple(map(int, args.size.split('x')))
    if len(size) != 2:
        raise ValueError("Size must be two integers separated by 'x'.")

    config = Config(
        source_path=args.source,
        output_path=args.output,
        size=size,
        grayscale=args.grayscale,
        image_format=args.format
    )

    processor = ImageProcessor(config)
    processor.process_images()
    processor.generate_csv()

if __name__ == '__main__':
    main()
