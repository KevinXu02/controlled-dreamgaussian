from PIL import Image
import os
from math import ceil, floor


def resize_and_fit_images(folder_path, output_path=None):
    # List all image files in the folder
    image_files = [
        f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Determine grid size based on the number of images
    num_images = len(image_files)
    if num_images in [4, 6, 8, 9]:
        grid_cols = int(ceil(num_images**0.5))
        grid_rows = int(ceil(num_images / grid_cols))
    else:
        raise ValueError("Folder must contain 4, 6, 8, or 9 images.")

    # Define the size for each individual image background
    image_bg_size = 256

    # Create a black background for the entire grid
    grid_bg = Image.new(
        "RGB", (image_bg_size * grid_cols, image_bg_size * grid_rows), (0, 0, 0)
    )

    for i, img_file in enumerate(image_files):
        # Open and resize image
        img_path = os.path.join(folder_path, img_file)
        with Image.open(img_path) as img:
            # Calculate new size while maintaining aspect ratio
            aspect_ratio = img.width / img.height
            if img.width > img.height:
                new_width = min(image_bg_size, img.width)
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = min(image_bg_size, img.height)
                new_width = int(new_height * aspect_ratio)

            # Resize image
            resized_img = img.resize((new_width, new_height))

            # Create an individual black background for each image
            individual_bg = Image.new("RGB", (image_bg_size, image_bg_size), (0, 0, 0))

            # Calculate position to paste in the individual background
            x = (image_bg_size - new_width) // 2
            y = (image_bg_size - new_height) // 2

            # Paste the resized image onto the individual background
            individual_bg.paste(resized_img, (x, y))

            # Calculate position to paste in the grid
            grid_x = (i % grid_cols) * image_bg_size
            grid_y = (i // grid_cols) * image_bg_size

            # Paste the individual background onto the grid
            grid_bg.paste(individual_bg, (grid_x, grid_y))

        # store the grid image
        if output_path:
            grid_bg.save(output_path)

    return grid_bg


def detach_images(grid_image, cols, rows):
    """
    Detach images from a grid image given the number of columns and rows.

    :param grid_image: PIL Image object of the grid.
    :param cols: Number of columns in the grid.
    :param rows: Number of rows in the grid.
    :return: List of PIL Image objects of the detached images.
    """
    # Calculate the size of each image
    image_width, image_height = grid_image.width // cols, grid_image.height // rows

    # List to hold detached images
    detached_images = []

    for row in range(rows):
        for col in range(cols):
            # Calculate the bounding box of the current image
            left = col * image_width
            upper = row * image_height
            right = left + image_width
            lower = upper + image_height

            # Crop the image and add to the list
            cropped_image = grid_image.crop((left, upper, right, lower))
            detached_images.append(cropped_image)

    return detached_images


# Example usage (assuming a grid image and cols, rows values)
# grid_image = final_grid_image  # This should be the PIL Image object of the grid
# cols, rows = 3, 3  # Adjust based on the grid layout
# images = detach_images(grid_image, cols, rows)
# for i, img in enumerate(images):
#     img.show()  # Display each detached image
#     # img.save(f'image_{i}.jpg')  # Optionally, save each image


# Example usage (assuming a folder path)
# folder_path = "path_to_your_folder"
# final_grid_image = resize_and_fit_images(folder_path)
# final_grid_image.show()  # Display the final grid image
