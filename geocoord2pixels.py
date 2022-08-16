import argparse
from osgeo import gdal


def world2pixels(geomatrix, x, y):
    """ Use a GDAL geomatrix to calculate the pixel offset from the top-left corner
    :param geomatrix: a GDAL geomatrix
    :param x: x coordinates
    :param y: y coordinates
    :return: x,y screen coordinates
    """

    ulx = geomatrix[0]  # topleft x
    uly = geomatrix[3]  # topleft y
    xdist = geomatrix[1]  # x pixel size
    ydist = geomatrix[5]  # y pixel size
    imgx = int((x - ulx) / xdist)  # image x coordinate
    imgy = int((y - uly) / ydist)  # image y coordinate

    return imgx, imgy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_pth", type=str, help="Path to the geotiff image file")
    parser.add_argument("--lat", type=float, help="Latitude - Y coordinate")
    parser.add_argument("--long", type=float, help="Longitude - X coordinate")
    

    args = parser.parse_args()
    spectral_vals = []

    try:
        # open the image
        geotiff = gdal.Open(args.image_pth)

        # getting geotransform info
        g = geotiff.GetGeoTransform()
        imgx, imgy = world2pixels(g, args.long, args.lat)

        for bnd in range(1, geotiff.RasterCount+1):
            rb = geotiff.GetRasterBand(bnd)
            spectral_vals.append(rb.ReadAsArray(imgx,imgy,1,1)[0][0])
            print(rb.ReadAsArray(imgx,imgy,1,1)[0][0])

    except Exception as err:
        print('Error: ', err)

    finally:
        # close the raster
        if geotiff is not None:
            geotiff = None

    
    return spectral_vals

if __name__ == "__main__":
    main()