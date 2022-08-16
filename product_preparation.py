# import
import rasterio as rio

import os
import sys
print(sys.executable)
import json
# import torch 
import torch
import logging
import rasterio
import numpy as np
from glob import glob
from glob import glob as glb
from rasterio.mask import mask
#from utils import predict_image
from model_scripts.models import CropClassifier
from model_scripts.model_utils import process_sample
from DatabaseConnector import call_update_data_and_product
from utils import reshape_img, CropClassifier, raster_merge_func, process_sample 


# modified launch_productprep
def launch_productprep(crop_type, model_path, data_dir, aoi_path, job_id, user_id, train_period): 
    """
    :param crop_type: str
    :param data_dir: str
    :param model_path: str
    :param train_period: int
    :param aoi_path: str
    :param job_id: str
    :param user_id: str
    :return predict_image_list, crop_raster_profile (rise_map image list, band count)
    """
    # Predicted Tiff List 
    Predicted_Tiff_list = [] 
    # Band Count 
    bc = None
    # increment value count no.of predicted maps
    incrmnt = 0
    map_file = None
    # predict_image_list list
    predict_image_list = []
    
    # define the processing directory and the VH, VV files to be used for model prediction
    processing_dir =  os.path.dirname(data_dir)
    product_dir = os.path.join(processing_dir, 'Products')
    print("product_dir : {}".format(product_dir))

    # define the output file location - check for existence
    os.makedirs(product_dir, exist_ok=True)
    out_file = os.path.join(product_dir, '{}_map.tif'.format(crop_type))
    print("Rise map file path  : {}".format(out_file))
    out_file_cropped = os.path.join(product_dir, '{}_map_cropped.tif'.format(crop_type))
    print("Cropped rise map file path  : {}".format(out_file_cropped))
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
    model.load_state_dict(torch.load(model_path))
    before_string_fitter = '{0}/{1}_*_VH*.tif'.format(data_dir, user_id)
    # remove double slash 
    after_string_fitter = before_string_fitter.replace("//","/")
    files = glb(after_string_fitter)
    print("files list : {}".format(files))
            
    for img_path in files:    
        incrmnt = incrmnt + 1
        print("Set - {}".format(incrmnt))
        print("img_path : {}".format(img_path))
        in_fileVH = img_path
        in_fileVV = img_path.replace('VH', 'VV')
        print("in_fileVV : {}".format(in_fileVV))
        
        # load and flatten images
        ts_vh_reshape, _ = reshape_img(in_fileVH, train_period)
        ts_vv_reshape, crop_raster_profile = reshape_img(in_fileVV, train_period)
        bc = crop_raster_profile
        
        # declare the product array to store the predicted results
        product_map = np.ones([crop_raster_profile['height'] * crop_raster_profile['width']])
        product_map = product_map.astype(np.float32)
        
        # model inference
        bs = 60000 # batch_size can be increased while using GPU
        count = 0
        while count < ts_vh_reshape.shape[1] - bs:
            ts_inp = np.concatenate([ts_vh_reshape[:, count:count + bs], ts_vv_reshape[:, count:count + bs]], axis=0)
            # send the transposed time-series so as to have (locations, ts)
            product_map[count:count + bs] = process_sample(model, ts_inp.transpose(1, 0), input_type='sample', return_prob=True)
            count += bs
            #print("while loop count : {}".format(count))
            # if (count % bs * 100) == 0:
            #     print(count)
        ts_inp = np.concatenate([ts_vh_reshape[:, count:], ts_vv_reshape[:, count:]], axis=0)
        product_map[count:] = process_sample(model, ts_inp.transpose(1, 0), input_type='sample', return_prob=True)
        product_map = product_map.reshape([crop_raster_profile['height'], crop_raster_profile['width']])
        map_file = os.path.join(product_dir, '{}_map.tif'.format(incrmnt))
        print('write product map {} starts...'.format(map_file))
        with rasterio.open(map_file, 'w', **crop_raster_profile) as dst:
            dst.write(product_map, 1)
        print('write product map {} completed!!!'.format(map_file))
        # map_file to Predicted_Tiff_list
        Predicted_Tiff_list.append(map_file)
        map_file = ''
        
    # raster_merge_product
    raster_merge_func(product_dir, Predicted_Tiff_list, out_file, out_file_cropped, aoi_path)
    
    # type conversion - String
    str_productMap_path = str(out_file)
    predict_image_list.append(str_productMap_path)
    predict_image_list.append(str_productMap_path)
    print("predict_image_list : {}".format(predict_image_list))
    # return predict_image_list, band count 
    return predict_image_list, bc    
    
if __name__ == "__main__":
    
    crop_type = 'rice'
    #model_path = '/home/ubuntu/Amuthan_weed_classification_venv/Lstm_Model_training_for_AgriX/Trained_Models/minmax-normalizer_lr0.01_savedmodel-1.pth'
    #model_path = '/home/ubuntu/Amuthan_weed_classification_venv/Lstm_Model_training_for_AgriX/Trained_Models/minmax-normalizer_lr0.001_savedmodel-1.pth'
    #model_path = '/home/ubuntu/mys3bucket/AgriX_Data/trained_models/Nizamabad_Rice_14dts/savedmodel.pth'
    model_path = '/home/ubuntu/mys3bucket/AgriX_Data/trained_models/Nizamabad_Rice_8dts/savedmodel.pth'
    #output_pth = '/home/ubuntu/mys3bucket/AgriX_Data/1/12293/Timeseries/' # nalgonda
    #output_pth = '/home/ubuntu/mys3bucket/AgriX_Data/1/12401/Timeseries/' # nizamabad
    output_pth = '/home/ubuntu/mys3bucket/AgriX_Data/1/12804/Timeseries/'
    aoi_path = '/home/ubuntu/Amuthan_weed_classification_venv/Data/Nizamabad.geojson'
    #job_id = '12401'
    job_id = '12804'
    user_id = '1'
    train_period = 12
    product_path, band_count = launch_productprep(crop_type, model_path, output_pth, aoi_path, job_id, user_id, train_period)
    print("product_path : {}".format(product_path))
    print("band_count : {}".format(band_count))
    
    '''
    crop_type = 'rice'
    #model_path = '/home/ubuntu/Amuthan_weed_classification_venv/Lstm_Model_training_for_AgriX/Trained_Models/minmax-normalizer_lr0.01_savedmodel-1.pth'
    model_path = '/home/ubuntu/Amuthan_weed_classification_venv/Lstm_Model_training_for_AgriX/Trained_Models/minmax-normalizer_lr0.001_savedmodel-1.pth'
    #model_path = '/home/ubuntu/mys3bucket/AgriX_Data/trained_models/Nizamabad_Rice_14dts/savedmodel.pth'
    #output_pth = '/home/ubuntu/mys3bucket/AgriX_Data/1/12293/Timeseries/' # nalgonda
    #output_pth = '/home/ubuntu/mys3bucket/AgriX_Data/1/12401/Timeseries/' # nizamabad
    output_pth = '/home/ubuntu/mys3bucket/AgriX_Data/1/12284/Timeseries/'
    #aoi_path = '/home/ubuntu/Amuthan_weed_classification_venv/Data/Nizamabad.geojson'
    aoi_path = '/home/ubuntu/Amuthan_weed_classification_venv/GeoJson_Files_for_test/polygon_0.06796697792248985.Geojson'
    job_id = '12284'
    user_id = '1'
    train_period = 12
    product_path, band_count = launch_productprep(crop_type, model_path, output_pth, aoi_path, job_id, user_id, train_period)
    print("product_path : {}".format(product_path))
    print("band_count : {}".format(band_count))
    '''
    '''
    # Product Raster Merge
    product_dir = '/home/ubuntu/mys3bucket/AgriX_Data/1/12401/Timeseries/Products/'
    Predicted_Tiff_list = ['/home/ubuntu/mys3bucket/AgriX_Data/1/12401/Timeseries/Products/1_map.tif', '/home/ubuntu/mys3bucket/AgriX_Data/1/12401/Timeseries/Products/2_map.tif']
    out_file = '/home/ubuntu/mys3bucket/AgriX_Data/1/12401/Timeseries/Products/rice_map.tif'
    out_file_cropped = '/home/ubuntu/mys3bucket/AgriX_Data/1/12401/Timeseries/Products/rice_map_cropped.tif'
    raster_merge_func(product_dir, Predicted_Tiff_list, out_file, out_file_cropped, aoi_path)
    '''