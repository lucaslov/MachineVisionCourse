#==================================================================================================
#                                               IMPORTS
#==================================================================================================
from __future__ import annotations
import os
import argparse
from PIL import Image
import pandas as pd

#==================================================================================================
#                                            IMAGE PROCESSING
#==================================================================================================
class ImageProcessor:
    '''
    A class for processing and organizing images based on given parameters.
    '''

    def __init__(self, source_path: str, output_path: str, size: tuple, grayscale: bool, format: str):
        '''
        Initializes an ImageProcessor instance.

        Args:
            source_path (str): The path where the input images are located.
            output_path (str): The path where processed images will be saved.
            size (tuple): The (width, height) to which images should be resized.
            grayscale (bool): Whether to convert images to grayscale.
            format (str): The output format of the images ('png' or 'jpg').
        '''
        self.source_path = source_path
        self.output_path = output_path
        self.size = size
        self.grayscale = grayscale
        self.format = format
        self.category_stats = []

    def process_images(self):
        '''
        Processes and organizes images into categories based on the configuration.
        '''
        # Read all images and organize them by category
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
                    # Build the paths
                    input_image_path = os.path.join(category_path, filename)
                    output_image_path = os.path.join(output_category_path, f'{processed_count:03d}.{self.format}')

                    # Open and process the image
                    image = Image.open(input_image_path)
                    if self.grayscale:
                        image = image.convert('L')
                    else:
                        image = image.convert('RGB')
                    image = image.resize(self.size)

                    # Save the image
                    image.save(output_image_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
                    abandoned_count += 1

            # Record the statistics for the category
            self.category_stats.append({
                'Category': category,
                'Number of images': processed_count,
                'Number of abandoned images': abandoned_count
            })

    def generate_csv(self):
        '''
        Generates a CSV file with the statistics of processed images.
        '''
        df = pd.DataFrame(self.category_stats)
        csv_path = os.path.join(self.output_path, 'image_statistics.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved statistics to {csv_path}')

#==================================================================================================
#                                            ARGUMENT PARSING
#==================================================================================================
def parse_arguments() -> argparse.Namespace:
    '''
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing parsed arguments.
    '''
    parser = argparse.ArgumentParser(description='Process and organize images into a dataset.')
    parser.add_argument('-s', '--source', type=str, required=True, help='Source directory containing images.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory for the processed images.')
    parser.add_argument('-sz', '--size', type=str, required=True, help='Output image size in WIDTHxHEIGHT format.')
    parser.add_argument('-g', '--grayscale', action='store_true', help='Convert images to grayscale.')
    parser.add_argument('-f', '--format', type=str, choices=['png', 'jpg'], required=True, help='Output image format.')
    return parser.parse_args()

#==================================================================================================
#                                             MAIN
#==================================================================================================
def main():
    '''
    Main function to orchestrate the image processing and organization.
    '''
    args = parse_arguments()
    size = tuple(map(int, args.size.split('x')))

    processor = ImageProcessor(args.source, args.output, size, args.grayscale, args.format)
    processor.process_images()
    processor.generate_csv()

if __name__ == '__main__':
    main()