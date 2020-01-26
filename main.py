from pathlib import Path

import cv2 as cv

import detector
import image

if __name__ == '__main__':
    # Load images from path
    images = image.get_by_dir(Path(r'./images/_source'))
    # Load classifiers from path
    classifiers = detector.get_classifiers(Path(r'./classifiers'))

    # Iterate through images
    for (path, img) in images:
        # Create the output path
        np = path.parents[1].joinpath('faces')
        # Get face coordinates in the image
        faces = detector.get_face_coordinates(img, classifiers)

        # Check if there are any faces in the image
        if faces is not None:
            # Create images from the face coordinates
            cimages = image.crop_multi(img, faces, 0.3)

            # Iterate over cropped images
            for (i, cimg) in enumerate(cimages):
                # Save the image into the output directory
                cv.imwrite(str(np.joinpath('%s-%d%s' % (path.stem, i + 1, path.suffix))), cimg)
