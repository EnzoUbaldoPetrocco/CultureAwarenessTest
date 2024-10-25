"""Module providing resized LAMP and CARPET datasets."""
__author__ = "Enzo Ubaldo Petrocco"
import sys
import pathlib
import cv2

sys.path.insert(1, "../../")

from Utils.FileManager.FileManager import FileManagerClass

class Preprocessing:
    """
    Preprocessing resizes the data to have the same size starting from original dataset
    """

    def create_ds(self, img_path, svpath, size):
        """
            This method gets images path and saves them with another size in svpath.
        """
        # create dir
        FileManagerClass(svpath)
        # get images from root
        types = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for typ in types:
            paths.extend(pathlib.Path(img_path).glob(typ))
        for i, pt in enumerate(paths):
            im = cv2.imread(str(pt))
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(svpath + f"im{i}.jpg", im)


def main():
    """
        Main function initialize all the paths of the datasets for building the datasets
    """
    prep = Preprocessing()
    base_pt = "../../../../FINALDS/"
    lampsize = 120
    carpetsize = 200

    chinoff = base_pt + "originals/lamps/chinese/off/"
    chinon = base_pt + "originals/lamps/chinese/on/"
    frenchoff = base_pt + "originals/lamps/french/off/"
    frenchon = base_pt + "originals/lamps/french/on/"
    turkoff = base_pt + "originals/lamps/turkish/off/"
    turkon = base_pt + "originals/lamps/turkish/on/"
    indoff = base_pt + "originals/carpets/indian/without/"
    indon = base_pt + "originals/carpets/indian/with/"
    japoff = base_pt + "originals/carpets/japanese/without/"
    japon = base_pt + "originals/carpets/japanese/with/"
    scanoff = base_pt + "originals/carpets/scandinavian/without/"
    scanon = base_pt + "originals/carpets/scandinavian/with/"

    svchinoff = base_pt + "/lamps/chinese/"
    svchinon = base_pt + "/lamps/chinese/"
    svfrenchoff = base_pt + "/lamps/french/"
    svfrenchon = base_pt + "/lamps/french/"
    svturkoff = base_pt + "/lamps/turkish/"
    svturkon = base_pt + "/lamps/turkish/"
    svindoff = base_pt + "/carpets_stretched/indian/"
    svindon = base_pt + "/carpets_stretched/indian/"
    svjapoff = base_pt + "/carpets_stretched/japanese/"
    svjapon = base_pt + "/carpets_stretched/japanese/"
    svscanoff = base_pt + "/carpets_stretched/scandinavian/"
    svscanon = base_pt + "/carpets_stretched/scandinavian/"

    prep.create_ds(chinoff, svchinoff + f"{lampsize}/RGB/off/", lampsize)
    prep.create_ds(chinon, svchinon + f"{lampsize}/RGB/on/", lampsize)
    prep.create_ds(frenchoff, svfrenchoff + f"{lampsize}/RGB/off/", lampsize)
    prep.create_ds(frenchon, svfrenchon + f"{lampsize}/RGB/on/", lampsize)
    prep.create_ds(turkoff, svturkoff + f"{lampsize}/RGB/off/", lampsize)
    prep.create_ds(turkon, svturkon + f"{lampsize}/RGB/on/", lampsize)
    prep.create_ds(indoff, svindoff + f"{carpetsize}/RGB/without/", carpetsize)
    prep.create_ds(indon, svindon + f"{carpetsize}/RGB/with/", carpetsize)
    prep.create_ds(japoff, svjapoff + f"{carpetsize}/RGB/without/", carpetsize)
    prep.create_ds(japon, svjapon + f"{carpetsize}/RGB/with/", carpetsize)
    prep.create_ds(scanoff, svscanoff + f"{carpetsize}/RGB/without/", carpetsize)
    prep.create_ds(scanon, svscanon + f"{carpetsize}/RGB/with/", carpetsize)


if __name__ == "__main__":
    main()
