import cv2
import os
import numpy as np
from pathlib import Path


def find_the_image(file_name, directory_name):
    """Search for the image file in the given directory and return its path."""
    for path, subdirs, files in os.walk(directory_name):
        if file_name in files:
            return os.path.join(path, file_name)

    print("Error: Image not found.")
    return None


# User inputs
image_name = input("Enter the name of the image file to process: ")
image_directory = input("Enter the directory containing the image: ")

# Find the image path
image_path = find_the_image(image_name, image_directory)

if image_path is None:
    exit()  # Exit if the image is not found

# Change working directory
os.chdir(Path(image_path).parent)

# Read the image
color_image = cv2.imread(image_path)

if color_image is None:
    print("Error: Unable to read the image. Please check the file format and path.")
    exit()

# Get user input for cartoon style
cartoon_style_selection = input("Choose a style (1: Smooth Painting, 2: Bold Sketchy): ")

if cartoon_style_selection == "1":
    # **Smooth, Painting-like Cartoon**
    cartoon_image = cv2.stylization(color_image, sigma_s=200, sigma_r=0.1)
    style = "cartoon_painting"

elif cartoon_style_selection == "2":
    # **Bold, Sketchy Cartoon**

    # Convert to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Apply median blur
    blurred = cv2.medianBlur(gray, 7)

    # Detect edges using adaptive threshold
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Apply bilateral filter for smooth color reduction
    color = cv2.bilateralFilter(color_image, d=9, sigmaColor=200, sigmaSpace=200)

    # Combine edges with color image
    cartoon_image = cv2.bitwise_and(color, color, mask=edges)
    style = "cartoon_sketch"

else:
    print("Invalid style selection.")
    exit()

# Show the cartoonified image
cv2.imshow(style, cartoon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

to_save = input("Do you want to save the image? (Y/N): ")
# Ask user where to save the image
if to_save == "Y":
    save_path = input(
        "Enter the directory to save the cartoonified image (Press Enter to save in the current directory): ").strip()

    # Default to the original image location if no path is provided
    if not save_path:
        save_path = Path(image_path).parent

    # Ensure the directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Construct save filename
    save_file = os.path.join(save_path, f"{Path(image_name).stem}_{style}.png")

    # Save the image
    cv2.imwrite(save_file, cartoon_image)
    print(f"Cartoonified image saved at: {save_file}")
else:
    print("No problem 😊")