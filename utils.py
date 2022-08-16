#### Raster Merge import libraries ####
#import os
import math
#import rasterio
#import numpy as np
from rasterio import Affine
from rasterio import windows
from tempfile import mkdtemp
#### Raster Merge import libraries ####
# Import
import io
import os
import glob
import shutil
### return ROI by using aoi_path ###
import ee
import numpy as np
import geopandas as gpd 
#### ####
import json
import torch
import rasterio
import torch.nn as nn
import earthpy.spatial as es
from glob import glob as glb
from rasterio.mask import mask
from apiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow  
#from google_drive_downloader import GoogleDriveDownloader as gdd
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
# Raster import 
import rasterio
import pandas as pd
from rasterio.merge import merge
# GDAl raster image merge
#import glob
import subprocess
from osgeo import gdal
# Roi read
ee.Initialize()

# Classes
class CropClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super(CropClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, batchX):
        out, (ht, ct) = self.lstm(batchX)
        return self.fc1(out[:, -1, :])

def stack_bands(input_dir, pol,job_id):
    """
    :param input_dir:
    :param pol:
    :return:
    """
    track_paths = glob(os.path.join(input_dir, 'processing', '*'))

    # create stacked time series for each track
    for track in track_paths:
        stack_band_paths = glob(os.path.join(track, 'Timeseries', '*{}.tif'.format(pol)))
        stack_band_paths.sort()

        output_dir = os.path.join(track, 'Timeseries', 'Timeseries_stacked_{}_{}.tif'.format(len(stack_band_paths), pol))
        raster_prof = es.stack(stack_band_paths, out_path=output_dir)
    
    print(f'Time series created.')

def stack_bands_rio(job_id,input_dir, dir='Timeseries'):
    """
    :param job_id:
    :param input_dir:
    :param dir:
    :return stacked_images, band_count
    """
    stacked_images = []
    band_count = 0
    track_paths = glob(os.path.join(input_dir, 'processing', '*'))

    # create stacked time series for each track
    for track in track_paths:
        # if stacked image already exists
        for pol in ['VH', 'VV']:
            if len(glob(os.path.join(track, dir, '{}_stacked_*_{}.tif'.format(dir, pol)))) != 0:
                output_pth = glob(os.path.join(track, dir, '{}_stacked_*_{}.tif'.format(dir, pol)))[0]
                stacked_images.append(output_pth)
                band_count = len(glob(os.path.join(track, dir, '*.bs.{}.tif'.format(pol))))
                continue

            stack_band_paths = glob(os.path.join(track, dir, '*{}.tif'.format(pol)))
            stack_band_paths.sort()
            band_count = len(stack_band_paths)

            # Read metadata of first file
            with rasterio.open(stack_band_paths[0]) as src0:
                meta = src0.meta

            # Update meta to reflect the number of layers
            meta.update(count=len(stack_band_paths))

            output_pth = os.path.join(track, dir,
                                    '{}_stacked_{}_{}.tif'.format(dir, len(stack_band_paths), pol))
            stacked_images.append(output_pth)
            # Read each layer and write it to stack
            with rasterio.open(output_pth, 'w', **meta) as dst:
                for id, layer in enumerate(stack_band_paths, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(id, src1.read(1))

        print(f'Time series created.')

    return stacked_images, band_count

# Google API Settings
class Settings(object):
    def __init__(self):
        self.__API_KEY = "AIzaSyArIwMoFtPRS1snoIyR4yt8Fam5Jtpqit8"
        
    @property
    def API_KEY(self):
        return self.__API_KEY

    @API_KEY.setter
    def API_KEY(self, value):
        self.__API_KEY = value
# Google API Methods
class GoogleDrive(Settings):
    def __init__(self):

        Settings.__init__(self)
        self.service = build("drive", "v3", developerKey=self.API_KEY)

    def get_files(self, folder_id=""):

        if folder_id == "":
            print("Folder ID cannot be None")
            return "Folder ID cannot be None"

        else:
            param = {
                "q": "'"
                + folder_id
                + "' in parents and mimeType != 'application/vnd.google-apps.folder'"
            }
            return [
                file
                for file in self.service.files().list(**param).execute().get("files")
            ]

    def download_file(self, file_id, mime_type="", file_name=""):

        request = self.service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False

        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))

        fh.seek(0)

        with open(file_name, "wb") as f:
            shutil.copyfileobj(fh, f, length=131072)
        return True

def get_bandcount(raster_path):
    """
    Function to get the number of bands in a raster image
    :param raster_path:
    :return bandcount
    """
    with rasterio.open(raster_path) as src:
        bandcount = src.profile['count']
    
    return bandcount
    
def roi_fromfile(aoi_path):
    """
    Function to read the geojson and return as EE ROI datatype
    If there are multiple polygons in the file, it merges them and gets the convex hull
    :param aoi_path:
    :return ROI
    """
    # Shape File Path
    gdf = gpd.read_file(aoi_path)
    gdf['const'] = 0 # add a constant column to combine the polygons based on this column value

    # combine the multi-record shapefile into a single polygon
    df_new = gdf.dissolve(by='const')
    dissolved_geom = [i for i in df_new.geometry]
    print("shape file length : {}".format(len(dissolved_geom)))

    #for Polygon geo data type
    x,y = dissolved_geom[0].exterior.coords.xy
    cords = np.dstack((x,y)).tolist()
    ROI=ee.Geometry.Polygon(cords)
    # return ROI
    return ROI
    
    
# load bands
def load_bands(file_path):
    """
    :param file_path:
    :return bands, crop_raster_profile
    """
    with rasterio.open(file_path) as src:
        bands = src.read()
        crop_raster_profile = src.profile

    crop_raster_profile['count'] = 1
    # return bands and crop raster profile
    return bands, crop_raster_profile


# used for reshape of image
def reshape_img(in_file, train_period):
    """
    :param in_file:
    :param train_period:
    :return ts_reshape, crop_raster_profile
    """
    # load the tif files
    image, crop_raster_profile = load_bands(in_file)

    # flatten the input bands to parallelize the predictions.
    ts_reshape = image.reshape([image.shape[0], image.shape[1] * image.shape[2]])

    # if the time period does not correspond to the range used for model training, add zero values.
    if ts_reshape.shape[0] < train_period:
        ts_reshape = np.append(ts_reshape,
                                  np.zeros([train_period - ts_reshape.shape[0], ts_reshape.shape[1]]), axis=0)
    elif ts_reshape.shape[0] > train_period:
        ts_reshape = ts_reshape[:train_period, :]
    
    # return ts_reshape, crop_raster_profile
    return ts_reshape, crop_raster_profile

# Used for product preparation
#def predict_image(data_dir, out_dir, model, train_period, aoi_path=None):
'''
def predict_image(crop_type, data_dir, model_path, train_period, aoi_path, job_id, user_id):
    """
    :param crop_type:
    :param data_dir:
    :param model_path:
    :param train_period:
    :param aoi_path:
    :param job_id:
    :param user_id:
    :return predict_image_list, crop_raster_profile (rise_map image list, band count)
    """
    # predict_image_list list
    predict_image_list = []
    
    # define the processing directory and the VH, VV files to be used for model prediction
    #processing_dir =  os.path.dirname(os.path.dirname(data_dir))
    processing_dir =  os.path.dirname(data_dir)
    product_dir = os.path.join(processing_dir, 'Products')
    print("product_dir : {}".format(product_dir))

    # define the output file location - check for existence
    os.makedirs(product_dir, exist_ok=True)
    out_file = os.path.join(product_dir, '{}_map.tif'.format(crop_type))
    #print("Rise map file path  : {}".format(out_file))
    out_file_cropped = os.path.join(product_dir, '{}_map_cropped.tif'.format(crop_type))
    #print("Cropped rise map file path  : {}".format(out_file_cropped))
    #out_file = os.path.join(out_dir, os.path.basename(img_path))
    #out_file_cropped = os.path.join(out_dir, 'crop'+os.path.basename(img_path))    
    band_count = 1
    print("defualt band count : {}".format(band_count))

    if os.path.exists(out_file_cropped):
      #loggerupds.info("INFO: PROD PREP: Job already processed: {}".format(out_file_cropped))
      print("INFO: PROD PREP: Job already processed: {}".format(out_file_cropped))
      print("Job already processed: ", out_file_cropped)
      return out_file_cropped, band_count

    # load the trained model and model related parameters
    # model_path = '/home/ubuntu/AgriX-Api/Crop_Analyser/models/rice_model.pth'
    print("model_path : {}".format(model_path))
    model = CropClassifier()    
    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path))
    #print("data_dir : {}".format(data_dir))
    #print("model_path : {}".format(model_path))
    before_string_fitter = '{0}/{1}_*_VH*.tif'.format(data_dir, user_id)
    #print("BEFORE string fitter applied : {}".format(before_string_fitter))
    # remove double slash 
    after_string_fitter = before_string_fitter.replace("//","/")
    #print("after_string_fitter : {}".format(after_string_fitter))
    files = glb(after_string_fitter)
    print("files list : {}".format(files))
            
    #for img_path in glob('{}/Sentinel1_VH*.tif'.format(data_dir)):
    for img_path in files:    
        print("img_path in_fileVH : {}".format(img_path))
        in_fileVH = img_path
        in_fileVV = '{}/{}'.format(data_dir, os.path.basename(img_path).replace('VH', 'VV'))
        print("in_fileVV : {}".format(in_fileVV))

        # load and flatten images
        ts_vh_reshape, _ = reshape_img(in_fileVH, train_period)
        #print("ts_vh_reshape : {}".format(ts_vh_reshape))
        ts_vv_reshape, crop_raster_profile = reshape_img(in_fileVV, train_period)
        #print("ts_vv_reshape and crop_raster_profile : {} and {}".format(ts_vh_reshape, crop_raster_profile))

        # declare the product array to store the predicted results
        product_map = np.ones([crop_raster_profile['height'] * crop_raster_profile['width']])
        #print("product_map np ones {}".format(product_map))
        product_map = product_map.astype(np.float32)
        #print("product_map astype : {}".format(product_map))

        # model inference
        bs = 60000 # batch_size can be increased while using GPU
        count = 0
        #print("while loop starts....")
        while count < ts_vh_reshape.shape[1] - bs:
            ts_inp = np.concatenate([ts_vh_reshape[:, count:count + bs], ts_vv_reshape[:, count:count + bs]], axis=0)
            #print("while loop ts_inp : {}".format(ts_inp))
            # send the transposed time-series so as to have (locations, ts)
            product_map[count:count + bs] = process_sample(model, ts_inp.transpose(1, 0), input_type='sample', return_prob=True)
            #print("while loop product_map[count:count + bs]  : {}".format(product_map[count:count + bs]))
            count += bs
            #print("while loop count : {}".format(count))
            # if (count % bs * 100) == 0:
            #     print(count)
        #print("while loop end.")
        ts_inp = np.concatenate([ts_vh_reshape[:, count:], ts_vv_reshape[:, count:]], axis=0)
        #print("ts_inp : {}".format(ts_inp))
        product_map[count:] = process_sample(model, ts_inp.transpose(1, 0), input_type='sample', return_prob=True)
        #print("product_map[count:count + bs]  : {}".format(product_map[count:]))
        product_map = product_map.reshape([crop_raster_profile['height'], crop_raster_profile['width']])
        #print("product_map reshape : {}".format(product_map))
        
        #print("write product map starts!!!!")
        #out_file = os.path.join(out_dir, os.path.basename(img_path))
        print('write product map {} starts...'.format(out_file))
        with rasterio.open(out_file, 'w', **crop_raster_profile) as dst:
            dst.write(product_map, 1)
        print('write product map {} completed!!!'.format(out_file))
        #out_file_cropped = os.path.join(out_dir, 'crop'+os.path.basename(img_path))
        
        with rasterio.open(out_file) as src:
        # crop the output product raster to the shape of aoi_json_file
            with open(aoi_path) as aoi:
                aoi_data = json.load(aoi)
            geoms = [feature["geometry"] for feature in aoi_data['features']]
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
        print('write cropped product map {} starts...'.format(out_file_cropped))
        with rasterio.open(out_file_cropped, "w", **out_meta) as dst:
            dst.write(out_image)
        print('write  cropped product map {} completed!!!'.format(out_file_cropped))

                   
    #print("completed!!!")
    str_out_file = str(out_file)
    predict_image_list.append(str_out_file)
    predict_image_list.append(str_out_file)
    print("predict_image_list : {}".format(predict_image_list))
    #return out_file_cropped, crop_raster_profile
    return predict_image_list, crop_raster_profile
'''
# Functions
# load all the grounth truth samples
def process_sample(model, fname, input_type='file', return_prob=False):
    """
    :param model:
    :param fname:
    :param input_type:
    :param return_prob:
    :return predict(model, tensorify(fname, input_type), input_type, return_prob=return_prob)
    """
    return predict(model, tensorify(fname, input_type), input_type, return_prob=return_prob)

def predict(model, sample, input_type, thresh=0.5, return_prob=False):
    """
    :param model:
    :param input_type:
    :param thresh:    
    :param return_prob:
    :return result
    """
    if input_type == 'file':
        sample, label = sample[0], sample[1]
        with torch.no_grad():
            ypred = torch.nn.functional.softmax(model(sample))
        result = ypred[:,1].tolist()
        # result = ypred.view(-1).tolist()[1]
        result = [1 if x >= thresh else 0 for x in result]
        acc, pre, rec, f1s, cfm = get_metrics(np.array(label), np.array(result))
        print('Acc:', acc, 'Prec:', pre, 'Rec:', rec, 'F1S:', f1s, 'CFM:', cfm)

    elif input_type == 'sample':
        with torch.no_grad():
            ypred = torch.nn.functional.softmax(model(sample))
        result = ypred[:, 1].tolist()
        # result = ypred.view(-1).tolist()[1]
        if return_prob:
            return result
        else:
            return [1 if x >= thresh else 0 for x in result]

def tensorify(data, input_type='file'):
    """
    :param data:
    :param input_type:
    :return features
    """
    if input_type == 'file':
        dfl = pd.read_csv(data)
        features = dfl.iloc[:, :-1]
        labels = dfl.iloc[:, -1]
        features, labels = torch.tensor(features.values).float(), torch.tensor(labels.values).long()
        # features = features.view(-1, 2, 15)
        features = features.view(-1, 2, features.shape[1] // 2)
        labels = labels.view(-1, 1)
        return features.transpose(2, 1), labels
    elif input_type == 'sample':
        features = torch.tensor(data).float()
        # features = features.view(-1, 2, 15)
        features = features.view(-1, 2, features.shape[1] // 2)
        features = features.transpose(2, 1)
        return features
    else:
        raise ValueError(f'input_type can either be "file", "sample". Cannot be {input_type}')
        
#def merge_rasters(raster_dir, search_crit, out_fp, aoi_path):
def merge_rasters(raster_dir, cropped_fp, search_crit, out_fp, aoi_path):
    """
    :param raster_dir:
    :param search_crit:
    :param out_fp:
    """
    # join file path with raster_dir and search_crit 
    q = os.path.join(raster_dir, search_crit)
    print("q : {}".format(q))
    rasters2merge =  glob.glob(q)
    # empty INPUT_FILES_List list
    INPUT_FILES_List = []
    for ras in rasters2merge:
      print("ras : {}".format(ras))
      INPUT_FILES_List.append(ras)
    print("raster merge function starts....")
    raster_merge_func(raster_dir, INPUT_FILES_List, out_fp, cropped_fp, aoi_path)  

def raster_merge_func(raster_dir, INPUT_FILES, timeseries_path, cropped_timeseries_path, aoi_path):
  """
   Merge raster with windowed r/w for memory efficient merging
  :param raster_dir: str
  :param INPUT_FILES: list
  :param timeseries_path: str
  :param cropped_timeseries_path: str
  :param aoi_path: str
  :return timeseries_path, - string
  """
  # crapped_timeseries_file initiation
  cropped_timeseries_file = ''
  
  # opened rasterio images where listed on input_files list
  sources = [rasterio.open(raster) for raster in INPUT_FILES]
  print("sources : {}".format(sources))
  
  # join test.mymemmap file with mkdtemp()
  #memmap_file = os.path.join(mkdtemp(), 'test.mymemmap')
  memmap_file = os.path.join(raster_dir, 'test.mymemmap')
  print("memmap_file path : {}".format(memmap_file))
  
  # adapted from https://github.com/mapbox/rasterio/blob/master/rasterio/merge.py
  first = sources[0]
  first_res = first.res
  dtype = first.dtypes[0]
  # Determine output band count
  output_count = first.count
  
  # bands list
  bands = [x + 1 for x in list(range(sources[0].count))]
  print("bands : {}".format(bands))
  
  # Extent of all inputs
  # scan input files
  xs = []
  ys = []
  for src in sources:
   left, bottom, right, top = src.bounds
   print("left : {}, bottom : {}, right : {}, top : {}".format(left, bottom, right, top))
   xs.extend([left, right])
   print("xs.extend([left, right]) : ".format(xs.extend([left, right])))
   ys.extend([bottom, top])
   print("ys.extend([bottom, top]) : {}".format(ys.extend([bottom, top])))
  dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)
  print("dst_w : {}, dst_s : {}, dst_e : {}, dst_n : {}".format(dst_w, dst_s, dst_e, dst_n))
  
  
  out_transform = Affine.translation(dst_w, dst_n)
  print("out_transform : {}".format(out_transform))
  
  # Resolution/pixel size
  res = first_res
  out_transform *= Affine.scale(res[0], -res[1])
  
  # Compute output array shape. We guarantee it will cover the output
  # bounds completely
  output_width = int(math.ceil((dst_e - dst_w) / res[0]))
  print("output_width : {}".format(output_width))
  output_height = int(math.ceil((dst_n - dst_s) / res[1]))
  print("output_height : {}".format(output_height))
  # Adjust bounds to fit
  dst_e, dst_s = out_transform * (output_width, output_height)
  print("Adjust bounds to fit -> dst_e : {}, dst_s : {}".format(dst_e, dst_s))
  # create destination array
  # destination array shape
  shape = (output_height, output_width)
  print("destination array shape  : {}".format(shape))
  #breakpoint()
  # dest = np.zeros((output_count, output_height, output_width), dtype=dtype)
  # Using numpy.memmap to create arrays directly mapped into a file
  dest_array = np.memmap(memmap_file, dtype=dtype,
                         mode='w+', shape=shape)
  
  dest_profile = {
      "driver": 'GTiff',
      "height": dest_array.shape[0],
      "width": dest_array.shape[1],
      "count": output_count,
      "dtype": dest_array.dtype,
      "crs": '+proj=latlong',
      "transform": out_transform,
      "nodata": 0
  }
  
  # open output file in write/read mode and fill with destination mosaick array
  with rasterio.open(
      os.path.join(timeseries_path),
      'w',
      **dest_profile
  ) as mosaic_raster:
      for src in sources:
          print("src : {}".format(src))
          for ID, b in enumerate(bands,1):
              print("ID and b : {} and {}".format(ID, b))
              for ji, src_window in src.block_windows(b):
                r = src.read(b, window=src_window)
                # store raster nodata value
                nodata = src.nodatavals[0]
                # using real world coordinates (bounds)
                src_bounds = windows.bounds(
                    src_window, transform=src.profile["transform"]) 
                #breakpoint()
                dst_window = windows.from_bounds(
                    *src_bounds, transform=mosaic_raster.profile["transform"])  
                #print("dst_window : {}".format(dst_window))  
                # round the values of dest_window as they can be float
                dst_window = windows.Window(round(dst_window.col_off), round(
                    dst_window.row_off), round(dst_window.width), round(dst_window.height))
                #mosaic_raster.write
                mosaic_raster.write(r, ID, window=dst_window)
            
  os.remove(memmap_file) 
  print("Raster File merge completed!!!")
  '''
  if aoi_path is not None:
    #out_file_cropped = os.path.join(out_dir, 'crop'+os.path.basename(img_path))
    with rasterio.open(os.path.join(timeseries_path)) as src:
        # crop the output product raster to the shape of aoi_json_file
        with open(aoi_path) as aoi:
            aoi_data = json.load(aoi)
        geoms = [feature["geometry"] for feature in aoi_data['features']]
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta
        #breakpoint()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
      
        with rasterio.open(cropped_timeseries_path, "w", **out_meta) as mosaic_raster:
            mosaic_raster.write(out_image)
  '''
    
  
if __name__ == "__main__":
        
    timeseries_file_path = '/home/ubuntu/mys3bucket/AgriX_Data/1/12061/Timeseries/'
    output_timeseries_path = '/home/ubuntu/mys3bucket/AgriX_Data/1/12061/Timeseries/Timeseries_stacked_VH_12061_windowmethod.tif'
    cropped_timeseries_path = '/home/ubuntu/mys3bucket/AgriX_Data/1/12061/Timeseries/cropped_Timeseries_stacked_VH_test.tif'
    search_crit = '1_12061_VH*.tif'
    aoi_path ="/home/ubuntu/Amuthan_weed_classification_venv/Amuthan_code/Nalgonda_epsg_4326.geojson"
    print("timeseries_file_path : {}".format(timeseries_file_path))
    print("output_timeseries_path : {}".format(output_timeseries_path))
    print("search_crit : {}".format(search_crit))
    merge_rasters(timeseries_file_path, cropped_timeseries_path, search_crit, output_timeseries_path, aoi_path)
    print("merge raster completed!!!")
    