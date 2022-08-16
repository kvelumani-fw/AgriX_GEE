# Import
import os
import sys
import time
import gdown # move tiff files from drive to local drive
import ee
import random # randomly choose Gdrive_folder, Gdrive_folder_id from Gdrive_folder_details dict
import logging
import rasterio
import numpy as np
from glob import glob
import geopandas as gpd
from pathlib import Path
import earthpy.spatial as es
from dataclasses import dataclass
# from medpy.filter import smoothing as S
from datetime import date, timedelta, datetime
from DatabaseConnector import call_update_data_and_product # always import in the end to avoid SSL certificate error with OST library
from utils import GoogleDrive, get_bandcount, merge_rasters
#from googleDriveFileDownloader import googleDriveFileDownloader # file url download
#from google_drive_downloader import GoogleDriveDownloader as gdd
# StaticData Class
@dataclass
class StaticData():
    #Polarisation_list
    Polarisation_list = ["VH", "VV"]
    s1_revisit = 12
    Gdrive_folder_details = {'AgriX-1':'1-nXpuRm71h2wWaK2iJ4WYuIc1NZZ1sB1', 'AgriX-2':'1PXvFUSVHghZLhnSG6dPWuFp4UURH-R44', 'AgriX-3':'1JvusSuviKbt34Pj6oSj5VWuc-bc_flSk', 'AgriX-4':'1wXivGvZCFzHNAxg_R-PhD-30WbU41P7i', 'AgriX-5':'1uF6ofLD-PSBasqJ058Qzl1dWV5JZ_9RO', 'AgriX-6':'17pNqbbrOthUsq_cprjrHdDyY2kEm9u5u'}
    #Gdrive_folder_details = {'AgriX-5':'1uF6ofLD-PSBasqJ058Qzl1dWV5JZ_9RO'}
    Gdrive_folder, Gdrive_folder_id = random.choice(list(Gdrive_folder_details.items()))
    print("randomly selected Gdrive_folder : {} \nGdrive_folder_id : {}".format(Gdrive_folder, Gdrive_folder_id))
    #local_bucket_path = Path('/home/ubuntu/earthengine/AgriX_Data')
    local_bucket_path = Path('/home/ubuntu/mys3bucket/AgriX_Data')
ee.Initialize()

def configure_procdirec(user_id, job_id):
    """
    :param user_id:
    :param job_id:
    :return:
    """
    try:
     # define a project directory
     home = Path.home()

     # create user_id folder and job_id folders as the processing directory
     home = StaticData.local_bucket_path
     # join user_id and job_id folders to home path
     project_dir = home.joinpath(user_id, job_id)
     # make directories
     os.makedirs(project_dir, exist_ok=True)
    except Exception as E:
        call_update_data_and_product(job_id,' ', ' ', '4','0')
        print(E)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
        print(E)
    return project_dir
     

# function to patchify and apply anisotropic diffusion when images are large
def patchify_asd(arr):
    try:
        # create cols, rows to split the arr
        div = 10000
        arr_new = np.zeros([arr.shape[0]+(div-arr.shape[0]%div), arr.shape[1]+(div-arr.shape[1]%div)], dtype=arr.dtype)
        arr_new[:arr.shape[0], :arr.shape[1]] = arr

        rows, cols = range(0, arr_new.shape[0], div), range(0, arr_new.shape[1], div)
        for r in range(1,len(rows)):
            for c in range(1,len(cols)):
                temp = arr_new[rows[r-1]:rows[r], cols[c-1]:cols[c]]
                # print(temp.shape, '{}:{}'.format(rows[r-1],rows[r]),'{}:{}'.format(cols[c-1],cols[c]))
                arr_new[rows[r-1]:rows[r], cols[c-1]:cols[c]] = S.anisotropic_diffusion(temp, niter=1, kappa=100, gamma=0.1, voxelspacing=np.array((3, 3)),
                                                 option=1)

        return arr_new[: arr.shape[0], :arr.shape[1]]

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
        print(E)


def anld_filter(input_dir):

    try:
        track_paths = glob(os.path.join(input_dir, 'processing', '*'))

        for track in track_paths:
            band_paths = glob(os.path.join(track, 'Timeseries', '*.tif'))
            band_paths.sort()
            # create a new folder to save ANLD filtered images
            out_dir = os.path.join(os.path.dirname(band_paths[0])+'_ANLD')
            if os.path.exists(out_dir):
                continue

            os.makedirs(out_dir, exist_ok=True)

            for band_path in band_paths:
                with rasterio.open(band_path) as src:
                    bands = src.read()
                    crop_raster_profile = src.profile
                    print(crop_raster_profile)
                    crop_raster_profile['count'] = bands.shape[0]
                    print("Number of bands to be processed:", bands.shape[0])

                temp = bands[0, :, :]
                bands[0, :, :] = patchify_asd(temp)

                out_file = os.path.join(out_dir, os.path.basename(band_path))
                with rasterio.open(out_file, 'w', **crop_raster_profile) as dst:
                    dst.write(bands)

        return out_dir

    except Exception as E:
        call_update_data_and_product(input_dir.parent.stem, ' ', ' ', '6', '0')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
        print(E)
        
'''
def roi_fromfile(aoi_path):
    """
    Function to read the geojson and return as EE ROI datatype
    If there are multiple polygons in the file, it merges them and gets the convex hull
    """
    # Shape File Path
    gdf = gpd.read_file(aoi_path)
    #print("gdf : {}".format(gdf))
    gdf['const'] = 0 # add a constant column to combine the polygons based on this column value

    # combine the multi-record shapefile into a single polygon
    df_new = gdf.dissolve(by='const')
    dissolved_geom = [i for i in df_new.geometry]
    print("shape file length : {}".format(len(dissolved_geom)))

    #for Polygon geo data type
    x,y = dissolved_geom[0].exterior.coords.xy
    cords = np.dstack((x,y)).tolist()
    ROI=ee.Geometry.Polygon(cords)

    return ROI
'''    
    
def roi_fromfile(aoi_path):
    """
    Function to read the geojson and return as EE ROI datatype
    If there are multiple polygons in the file, it merges them and gets the convex hull
    """
    # Shape File Path
    gdf = gpd.read_file(aoi_path)
    #print("gdf : {}".format(gdf))
    gdf['const'] = 0 # add a constant column to combine the polygons based on this column value

    # combine the multi-record shapefile into a single polygon
    df_new = gdf.dissolve(by='const')
    dissolved_geom = [i for i in df_new.geometry]
    print("shape file length : {}".format(len(dissolved_geom)))

    # for Polygon geo data type
    x,y = dissolved_geom[0].exterior.coords.xy
    cords = np.dstack((x,y)).tolist()
    ROI=ee.Geometry.Polygon(cords)

    return ROI


def GDrive_2_Localpath(Gdrive_folder_id, user_id, job_id, gdfilename):
  """
  :param Gdrive_folder_id:
  :param user_id:
  :param job_id:
  :param gdfilename:
  :return:
  downloaded_image_bucket_path returntype str location where the downloaded tiff images are stored
  """
  #create object for gooogleDrive Class
  helper = GoogleDrive()
  # define drive folder_id
  files = helper.get_files(folder_id=Gdrive_folder_id)
  print("files : {}".format(files))  
  # load root directory  
  home_dir = StaticData.local_bucket_path
  # create sub-directory using user_id, job_id, "Timeseries"
  downloaded_image_bucket_path = home_dir.joinpath(user_id, job_id, "Timeseries")
  downloaded_image_bucket_path = downloaded_image_bucket_path
  # call make directory
  os.makedirs(downloaded_image_bucket_path, exist_ok=True)
  # print path 
  print("Local Bucket Path : {}".format(downloaded_image_bucket_path))
  print("files : {}".format(files))
  # load folder id details using for loop
  for file in files:
    # assign file name on temp variable
    name = file.get("name")    
    # print file name
    print("File Name and Job_Id : {} and {}".format(name, job_id))  
    # check the file name which contains the user_id 
    #if name.find(job_id) == -1: #gdfilename
    if name.find(gdfilename) == -1:
      # print condition fails
      print("No file name {} is here!".format(gdfilename)) 
      # make name variable as a null
      name = ''       
    else:
      print("Expected File name is Present on the drive!!!")
      print("File name : {}".format(name))
      # join filename with destination path
      FileDestPath = downloaded_image_bucket_path.joinpath(file.get("name"))
      FileDestPath = str(FileDestPath)
      print("FileDestPath on bucket: {}".format(FileDestPath))
      # download the file using a.downloadFile Method and add the fil_id to the url         
      url = "https://drive.google.com/uc?id={}".format(file.get("id"))
      #print url
      print("URL : {}".format(url))
      # call url download fuinction for download the drive image
      gdown.download(url, FileDestPath, quiet=False) # download both small and larger files           
      # make name variable as a null
      name = ''   
      # ackonwledge files download successfully         
      print("Files moved successfully to mys3bucket!!!")
  # return file downloaded destination path 
  return downloaded_image_bucket_path

# 20/02/2022(DD/MM/YYYY) - Modified DataPreparation Function
def launch_dataprep(user_id,job_id,start_date,end_date, aoi_path, product_type, direction,
                    ls_mask, rmv_speckle, resolution):
    """
    :param user_id:
    :param job_id:
    :param start_date:
    :param end_date:
    :param aoi_path:
    :param product_type:
    :param direction:
    :param ls_mask:
    :param rmv_speckle:
    :param resolution:
    :return:
    stacked_path returntype str location where the final stacked image is stored
    """
    
    try:
      
      project_dir = configure_procdirec(user_id, job_id)
      loggerupds = logging.getLogger('update') 
      # get ROI from file 
      print("select geojson or shp file : {}".format(aoi_path))
      ROI = roi_fromfile(aoi_path)
      Polarisation_list = StaticData.Polarisation_list
      revisit = StaticData.s1_revisit
      
      # Empty list declaratation
      stacked_path_list = []  
      crapped_timeseries_file_list = []
      
      print("start_date {} and end_date : {}".format(start_date, end_date))
      loggerupds.info("randomly selected Gdrive_folder : {} \nGdrive_folder_id : {}".format(StaticData.Gdrive_folder, StaticData.Gdrive_folder_id))
      #print("randomly selected Gdrive_folder : {} \nGdrive_folder_id : {}".format(StaticData.Gdrive_folder, StaticData.Gdrive_folder_id))
      for requiredPolarisation in Polarisation_list:
        #print("select based on polarization : {}".format(requiredPolarisation))  
        # create image collection 
        imageCollection = ee.ImageCollection('COPERNICUS/S1_GRD') \
          .filter(ee.Filter.eq('instrumentMode', 'IW')) \
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
          .filter(ee.Filter.eq('orbitProperties_pass', direction.split("_")[0])) \
          .filterBounds(ROI) \
          .filter(ee.Filter.date(start_date, end_date))       
        # filter only polarization required for download        
        imageCollection = imageCollection.select(requiredPolarisation) 
        stack_li = []
        start_date_dateformat_convertion = datetime.fromisoformat(start_date)
        end_date_dateformat_convertion = datetime.fromisoformat(end_date)
  
        delta = timedelta(days=revisit)
        count = 0
        deltaadd = timedelta(days=revisit-1)
        original_end_date = start_date_dateformat_convertion + deltaadd
        original_end_date_delta = timedelta(days=revisit)
        addOneDate = timedelta(days=1)  
          
        # merge tiles taken during a period of 12 days (s1 revisit period)
        while start_date_dateformat_convertion <= end_date_dateformat_convertion:
          from_date = start_date_dateformat_convertion
          t1 = start_date_dateformat_convertion.strftime("%Y-%m-%d")
          start_date_dateformat_convertion += delta 
          to_date = original_end_date
          t2 = original_end_date.strftime("%Y-%m-%d") 
          original_end_date += original_end_date_delta
          if start_date_dateformat_convertion <= end_date_dateformat_convertion:
            print("Band range : Start Date : {}, End Date : {}".format(t1, t2))
            # get imagecollection images
            image_filt = imageCollection.filterDate(t1, t2)
            image_filt = image_filt.median().clip(ROI)
            # append imagecollection to stack_li list
            stack_li.append(image_filt)
          else:
            print("out of the range, Start Date : {}, End Date : {}".format(t1, t2))
            break     
        # Concadinate the listed image     
        composite = ee.Image.cat([x for x in stack_li])  
        #loggerupds.info(f'INFO: DATA PREP: Processing and creating ARD time-series')
        loggerupds.info('INFO: DATA PREP: Processing and creating ARD time-series')
        #print('File download Starts....')
        # fileNamePrefix pattern
        #gdfilename = f'{user_id}_{job_id}_{requiredPolarisation}'
        gdfilename = '{}_{}_{}'.format(user_id, job_id, requiredPolarisation)
        # download the image from google earth engine to google drive
        task=ee.batch.Export.image.toDrive(composite, folder=StaticData.Gdrive_folder, 
          fileNamePrefix=gdfilename, scale=resolution, region=ROI,  maxPixels =1e13)       
        # Download dataset to GoogleDrive
        task.start() 
        # get status of the task
        status = task.status()
        while (status['state'] == 'READY') | (status['state'] == 'RUNNING'):
          time.sleep(3)
          status = task.status()
        # task completed
        if status['state'] == 'COMPLETED':
          print(status['state'])
          print("File Download {}!!".format(status['state']))                      
          # Download from GDrive to Local Home Folder
          timeseries_file_path = GDrive_2_Localpath(StaticData.Gdrive_folder_id, user_id, job_id, gdfilename)
          print("timeseries file path : {}".format(timeseries_file_path))
          timeseries_path = timeseries_file_path.joinpath("Timeseries_stacked_{}.tif".format(requiredPolarisation))
          #print("output timeseries path : {}".format(timeseries_path))
          # search_crit patterns
          search_crit = "{}_{}_{}*.tif".format(user_id, job_id, requiredPolarisation)
          #print("search_crit : {}".format(search_crit))
          #print("merge_rasters starts!!!!")
          
          # Merge rasters 
          ### old function ###
          #merge_rasters(timeseries_file_path, search_crit, timeseries_path) #aoi_path
          ### - ###
          ### new function ###
          crapped_timeseries_path = timeseries_file_path.joinpath("crapped_Timeseries_stacked_{}.tif".format(requiredPolarisation))
          print("crapped_timeseries_path : {}".format(crapped_timeseries_path))
          #crapped_timeseries_file = merge_rasters(timeseries_file_path, crapped_timeseries_path, search_crit, timeseries_path, aoi_path)
          merge_rasters(timeseries_file_path, crapped_timeseries_path, search_crit, timeseries_path, aoi_path)
          #crapped_timeseries_file = crapped_timeseries_path
          print("crapped_timeseries_file : {}".format(crapped_timeseries_path))
          str_crapped_timeseries_path = str(crapped_timeseries_path)
          crapped_timeseries_file_list.append(str_crapped_timeseries_path)
          
          loggerupds.info("INFO: DATA PREP: Stacking the time-series into single raster")           
          #print("merge_rasters Completed!!!!")
          # convert the timeseries_path posixpath to str
          str_timeseries_path = str(timeseries_path)
          # stack the VV and VH images on stacked_path_list list
          stacked_path_list.append(str_timeseries_path)
          #print("timeseries_path : {} was appended to stacked_path_list list".format(stacked_path_list))
          # Get Image band count
          band_count = get_bandcount(stacked_path_list[0])
          #print("Band Count : {}".format(band_count))
          timeseries_file_path = str(timeseries_file_path)
          loggerupds.info("INFO: DATA PREP: Data prep finished")
        # task failed
        elif status['state'] == 'FAILED':
          print(status['state'])
          print("File Download {}!!".format(status['state']))         
      # return timeseries_file_path, stacked_path_list, band_count
      ####return timeseries_file_path, stacked_path_list, band_count # old function
      #return timeseries_file_path, crapped_timeseries_file_list, band_count
      return timeseries_file_path, stacked_path_list, band_count
          
    except Exception as E:
        call_update_data_and_product(job_id,' ',' ','9','0')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
        print(E)

        print(E)

if __name__ == "__main__":
    '''
    # Test GDrive_2_Localpath function
    user_id='21'
    job_id='643015'
    gdfilename = '21_643015_VH'
    downloaded_image_bucket_path = GDrive_2_Localpath(StaticData.Gdrive_folder_id, user_id, job_id, gdfilename)
    print("downloaded_image_bucket_path : {}".format(downloaded_image_bucket_path)) 
    '''
    
    # To Test roi_fromfile(aoi_path)
    #aoi_path = '/home/ubuntu/AgriX-Api/AgriXApi-Release/GeoJson/polygon_0.6678702589228258.Geojson' # exterior errored geojson
    #aoi_path = '/home/ubuntu/Amuthan_weed_classification_venv/Data/Nizamabad_New_District_Shape_File/Nizamabad.shp' # nizamabad shp file
    #aoi_path = '/home/ubuntu/Amuthan_weed_classification_venv/Data/polygon_0.06796697792248985.Geojson' # sample geojson file
    #aoi_path = '/home/ubuntu/Amuthan_weed_classification_venv/GeoJson_Files_for_test/Nalgonda.Geojson' #  Nalgonda Geojson
    #aoi_path = '/home/ubuntu/Amuthan_weed_classification_venv/GeoJson_Files_for_test/Nizamabad.geojson' #  Nizamabad Geojson
    #aoi_path = "/home/ubuntu/mys3bucket/RiceMap_GEE/ROI/TSDM/Mandal_Boundary.shp" # telangana state
    
        
    #roi_fromfile(aoi_path)
    #print("aoi_path : {}".format(aoi_path))
    #ROI = roi_fromfile(aoi_path)
    #print("ROI : {}".format(ROI))
    
    # Test GDrive_2_Localpath function
    user_id='1'
    job_id='12427'
    gdfilename = '1_12427_VH'
    #Gd_id = '1uF6ofLD-PSBasqJ058Qzl1dWV5JZ_9RO' # agrix-5
    Gd_id = '1PXvFUSVHghZLhnSG6dPWuFp4UURH-R44'
    downloaded_image_bucket_path = GDrive_2_Localpath(Gd_id, user_id, job_id, gdfilename)
    print("downloaded_image_bucket_path : {}".format(downloaded_image_bucket_path)) 
    