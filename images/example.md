# Adding Images to the Dataset

This directory (`images/`) is where you should place your 16x16 pixel art images for training the LunarCoreVAE model. The model will automatically detect and use any JPEG, PNG, or JPG images added to this folder.

## Image Requirements:

*   **Format:**  JPEG, PNG, or JPG (`.jpg`, `.jpeg`, `.png`)
*   **Size:** 16x16 pixels
*   **Content:** Pixel art
*   **Color Mode:** RGB (make sure your image has 3 color channels, even if it is grayscale)

## How to Add Images:

1.  **Prepare your pixel art images:** Make sure your images meet the requirements listed above.
2.  **Name your images:** You can name your images anything you like, but it is recommended to use a consistent naming scheme (for example `image_001.jpg`, `image_002.jpg`, etc.). The `.jpg`, `.jpeg`, or `.png` extensions must be lowercase.
3.  **Place the images in the `images/` directory:** Simply copy or move your 16x16 pixel art images into the `images/` folder.

## Training:

Once you have added your images, you can start the training process. The training script (`train_lunar_core.py`) will automatically load all JPEG, PNG and JPG images from this directory.

## Notes:

*   The more diverse and high-quality your dataset is, the better the model will learn.
*   You do not need to add a separate file with labels if you are simply training the VAE for image reconstruction.
*   Feel free to experiment with different types of pixel art to see how it affects the model's output.

**Example:**

If you have a 16x16 pixel art image named `my_pixel_art.png`, simply place it inside the `images/` directory. The training script will automatically use it during the next training run.

**Have fun training your LunarCoreVAE model!**
