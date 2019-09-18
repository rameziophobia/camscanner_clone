HOW TO USE:

    1. open the terminal
    2. run command: python camscanner.py --image [IMG_PATH] --lang [LANG]
    3. replace IMG_PATH with the relative or abs. path to the img, replace
       LANG with your language of choice ex: ara, eng.
       notes: tesseract training data should be installed on the system
              if on linux or tesseract is added to path comment line 7
    4. upon running the input image and the final img will be shown if
       the orientation is not correct press 'a' or 'd' to rotate 90 degrees
    5. press "space" to continue
    6. two text files will be created output.txt and noisyImages_output.txt
       they will include the text in the image with each doing its own filters
       use the file with the best results for each image