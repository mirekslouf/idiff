import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import stemdiff as sd
import pandas as pd
import dbase
import os


class IntensityAnalyzer:
    def __init__(self, img_path, df_path, num_images=5):
        self.img_path = img_path
        self.df_path = df_path
        self.num_images = num_images
        self.intensity_ratios = []

        # Initialize source data
        self.SDATA = sd.gvars.SourceData(
            detector=sd.detectors.TimePix(),
            data_dir=img_path,
            filenames=r'*.dat'
        )

        # Load dataframe
        self.df_sum = dbase.read_database(df_path)
        self.df_sum = self.df_sum.sort_values(by='S', ascending=False)

    def load_image(self, index):
        """Load image with specified index from sorted dataframe"""
        datafile = self.df_sum.iloc[index]['DatafileName']
        datafile_name = self.SDATA.data_dir.joinpath(datafile)
        arr = sd.io.Datafiles.read(self.SDATA, datafile_name)
        arr = arr.astype(np.float32)
        return arr, datafile

    def get_box_selector(self, ax, callback):
        """Create a rectangle selector on the given axis"""
        rect_selector = RectangleSelector(
            ax, callback,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        return rect_selector

    def calculate_box_intensity(self, img, box):
        """Calculate average intensity in the given box"""
        x1, y1, x2, y2 = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Convert to integers for slicing
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Extract the region and calculate average
        region = img[y1:y2, x1:x2]
        avg_intensity = np.mean(region)

        return avg_intensity

    def analyze_image(self, img, filename):
        """Process a single image, collecting bounding boxes and calculating intensities"""
        # Set up figure for display
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display the image with a logarithmic scale for better visibility
        im = ax.imshow(np.log1p(img), cmap='viridis')
        plt.colorbar(im, ax=ax, label='Log Intensity')
        ax.set_title(f"Image: {filename}\nSelect central peak region")

        # Variables to store box coordinates and intensities
        boxes = []
        central_intensity = None
        radial_intensities = []

        # Callback function for box selection
        def on_select(eclick, erelease):
            nonlocal boxes
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            boxes.append((x1, y1, x2, y2))

            # Draw the rectangle on the plot
            rect = patches.Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1), abs(y2 - y1),
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            plt.draw()

        # First, select the central peak
        rs = self.get_box_selector(ax, on_select)
        plt.tight_layout()
        plt.show()

        # Calculate central peak intensity
        if boxes:
            central_intensity = self.calculate_box_intensity(img, boxes[0])
            print(f"Central peak average intensity: {central_intensity:.2f}")

            # Now select the 5 radial peaks
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(np.log1p(img), cmap='viridis')
            plt.colorbar(im, ax=ax, label='Log Intensity')

            # Add the central peak box to the new plot
            x1, y1, x2, y2 = boxes[0]
            rect = patches.Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1), abs(y2 - y1),
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            ax.set_title(f"Image: {filename}\nSelect 5 radial peak regions")
            boxes = []  # Reset boxes for radial peaks

            rs = self.get_box_selector(ax, on_select)
            plt.tight_layout()
            plt.show()

            # Calculate radial peak intensities
            for i, box in enumerate(boxes):
                intensity = self.calculate_box_intensity(img, box)
                radial_intensities.append(intensity)
                print(f"Radial peak {i + 1} average intensity: {intensity:.2f}")

            # Calculate average and ratio
            if radial_intensities:
                avg_radial_intensity = np.mean(radial_intensities)
                ratio = avg_radial_intensity / central_intensity
                print(f"Intensity ratio (radial/central): {ratio:.4f}")
                return ratio

        return None

    def run_analysis(self):
        """Run the analysis on the top N images by S value"""
        for i in range(min(self.num_images, len(self.df_sum))):
            print(f"\nAnalyzing image {i + 1} of {self.num_images}")
            img, filename = self.load_image(i)
            ratio = self.analyze_image(img, filename)

            if ratio is not None:
                self.intensity_ratios.append(ratio)

        # Calculate and save the average ratio
        if self.intensity_ratios:
            avg_ratio = np.mean(self.intensity_ratios)
            std_ratio = np.std(self.intensity_ratios)

            print(f"\nAnalysis complete!")
            print(f"Average intensity ratio: {avg_ratio:.4f} ± {std_ratio:.4f}")

            # Save results
            with open('intensity_ratio_results.txt', 'w') as f:
                f.write(f"Analysis of {len(self.intensity_ratios)} images\n")
                f.write(f"Average intensity ratio (radial/central): {avg_ratio:.4f}\n")
                f.write(f"Standard deviation: {std_ratio:.4f}\n\n")

                f.write("Individual image results:\n")
                for i, ratio in enumerate(self.intensity_ratios):
                    f.write(f"Image {i + 1}: {ratio:.4f}\n")

            print(f"Results saved to 'intensity_ratio_results.txt'")


# Main execution
if __name__ == "__main__":
    # Set paths
    img_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA"
    df_path = r"C:\Users\drend\OneDrive\Plocha\VU\1_AU\1_AU\DATA\resultsdbase_sum.zip"

    # Create analyzer and run
    analyzer = IntensityAnalyzer(img_path, df_path, num_images=5)
    analyzer.run_analysis()