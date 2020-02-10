from pathlib import Path

import cv2 as cv

import detector
import image

if __name__ == '__main__':
    # Načti obrázky z adresáře
    images = image.get_by_dir(Path(r'./images/_source'))
    # Načti klasifikátory z adresáře
    classifiers = detector.get_classifiers(Path(r'./classifiers'))

    # Projdi všechny načtené obrázky
    for (path, img) in images:
        # Sestav cestu pro uložení
        np = path.parents[1].joinpath('faces')
        # Získej souřadnice obličejů na obrázku
        faces = detector.get_face_coordinates(img, classifiers)

        # Zkontroluj, jestli byl nalezen alespoň 1 obličej
        if faces is not None:
            # Vytvoř obrázky ze získaných souřadnic obličejů
            cimages = image.crop_multi(img, faces, 0.3)

            # Projdi jednotlivé oříznuté obrázky
            for (i, cimg) in enumerate(cimages):
                # Ulož obrázek do výstupního adresáře
                cv.imwrite(str(np.joinpath('%s-%d%s' % (path.stem, i + 1, path.suffix))), cimg)
