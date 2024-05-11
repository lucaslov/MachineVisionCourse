"""
This module processes and organizes images into a dataset based on user-defined parameters.
It supports resizing, optional grayscale conversion, and categorization of images.
"""

import os
import argparse
from PIL import Image
import pandas as pd

class ImageProcessor:
    """
    A class for processing and organizing images based on given parameters.
    """

    def __init__(self, config):
        """
        Initializes an ImageProcessor instance using a configuration object.

        Args:
            config (object): Configuration object containing source_path, output_path,
                             size, grayscale, and image_format.
        """
        self.source_path = config.source_path
        self.output_path = config.output_path
        self.size = config.size
        self.grayscale = config.grayscale
        self.image_format = config.image_format
        self.category_stats = []

    def process_images(self):
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
                'Category': category,
                'Number of images': processed_count,
                'Number of abandoned images': abandoned_count
            })

    def generate_csv(self):
        """
        Generates a CSV file with the statistics of processed images.
        """
        df = pd.DataFrame(self.category_stats)
        csv_path = os.path.join(self.output_path, 'image_statistics.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved statistics to {csv_path}')

def parse_arguments():
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
    parser.add_argument('-f', '--format', type=str, choices=['png', 'jpg'], \
        required=True, help='Output image format.')
    return parser.parse_args()

def main():
    """
    Main function to orchestrate the image processing and organization.
    """
    args = parse_arguments()
    size = tuple(map(int, args.size.split('x')))

    # Create a configuration object
    config = argparse.Namespace(
        source_path=args.source,
        output_path=args.output,
        size=size,
        grayscale=args.grayscale,
        image_format=args.format
    )

    processor = ImageProcessor(config)
    processor.process_images()
    processor.generate_csv()

# Ensure the file ends with a newline
if __name__ == '__main__':
    main()
