import logging
import os
import sys
import pandas as pd
from glob import glob
import datetime
import numpy as np
import math

import torch
import torch.nn as nn

from geojson import Feature, FeatureCollection, Point
import fiona
import geopandas
from fiona.crs import from_epsg
from shapely.geometry import MultiPoint, shape, mapping
import rasterio as rio

from model_scripts.data_utils import CropDataset
from model_scripts.models import CropClassifier
from model_scripts.model_utils import get_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# StandardScaler
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# ignore warnings
import warnings
warnings.simplefilter("ignore")

from data_preparation import launch_dataprep, configure_procdirec
from DatabaseConnector import call_update_modaltrainning
loggerupds = logging.getLogger('update')

def read_label_file(label_file_path, clabel, crop_name):
    try:

        # read excel
        label_file = pd.read_excel(label_file_path, engine='openpyxl')

        # create a new column 'label_id' to store the 0, 1 values
        label_file['label_id'] = 0
        label_file.loc[label_file[clabel] == crop_name, 'label_id'] = 1

        label_file.rename(columns={'Longitude': 'X', 'Latitude': 'Y'}, inplace=True)

        return label_file

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))


def excel_to_geojson(label_file_path, clat, clong, clabel):
    # convert excel to a point geojson format
    try:
        # check file extension
        _, extension = os.path.splitext(label_file_path)
        if extension.lower() not in ['.xls', '.xlsx']:
            loggerupds.info('ERROR: The label file is not in Excel format. Please pass an excel file')
            sys.exit(-1)

        df = pd.read_excel(label_file_path, engine='openpyxl')
        df = df.dropna(subset=[clat, clong, clabel], how='all')
        features = []

        for ind, row in df.iterrows():
            latitude, longitude = map(float, (row[clat], row[clong]))
            features.append(
                Feature(
                    geometry=Point((longitude, latitude)),

                    properties={
                        'crop_name': row[clabel]
                    }
                )
            )

        collection = FeatureCollection(features)
        return collection

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))


def save_shapefile(collection, fname, geometry="Point"):
    """
    Save one `geometry` type from a geojson of a __geo_interface__ as a
    shapefile`.

    CAVEAT: this is a lossy conversion! I am passing along only the name
    property.

    """
    try:
        schema = {"geometry": geometry,
                  "properties": {"crop_name": "str:80"}}

        with fiona.open(fname, "w", "ESRI Shapefile", schema, crs=from_epsg(4326)) as f:
            for k, feature in enumerate(collection["features"]):
                if feature["geometry"]["type"] == geometry:
                    try:
                        crop_name = feature["properties"]["crop_name"]
                    except KeyError:
                        crop_name = k
                    f.write({
                        "geometry": feature["geometry"],
                        "properties": {"crop_name": crop_name},
                        })

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))


def pt2polygon_convexhull(ptshp_path, polyshp_path, buffer_r=0.1):
    """
    Read the points shapefile and create a minimum area bounding box
    :param ptshp_path: Path to the point shape file
    :param aoi_path: Output path to save the boudning box shape file
    :param buffer: Buffer radius
    :return:
    """
    try:
        mpt = MultiPoint([shape(point['geometry']) for point in fiona.open(ptshp_path)])
        geom = mpt.convex_hull  # the shapely geometry
        schema = {'geometry': 'Polygon', 'properties': {'name': 'str'}}
        with fiona.open(polyshp_path, 'w', 'ESRI Shapefile', schema, crs=from_epsg(4326)) as output:
            output.write({'properties': {'name': '0'}, 'geometry': mapping(geom.buffer(buffer_r))})

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))


def latlong2rowcol(input_file, X, Y):
    with rio.open(input_file) as src:
        bands = src.read()

    row, col = src.index(X, Y)
    return row, col

def load_bands(file_path):
    with rio.open(file_path) as src:
        bands = src.read()
        crop_raster_profile = src.profile

    crop_raster_profile['count'] = 1
    return bands, crop_raster_profile

def get_imagedn(in_file, row, col, exp_bands):
    image, _ = load_bands(in_file)
    dnvalue = image[:, row, col].transpose()

    # if the no. of. bands is less than period*3, add zero values
    if dnvalue.shape[-1] < exp_bands:
        dnvalue = np.append(dnvalue, np.zeros((dnvalue.shape[0], exp_bands - dnvalue.shape[1])), axis=1)
    elif dnvalue.shape[-1] > exp_bands:
        dnvalue = dnvalue[:, :exp_bands]
    return dnvalue

'''
def latlong2DN(label_file, in_fileVH, in_fileVV, period):
    try:          
        exp_bands = math.ceil((period*30)/12)
    
        row, col, l = latlong2rowcol(in_fileVH, label_file['X'], label_file['Y'], label_file['label_id'])
        dnvalueVH = get_imagedn(in_fileVH, row, col, exp_bands)
        dnvalueVV= get_imagedn(in_fileVV, row, col, exp_bands)
    
        dnvalues = np.concatenate([dnvalueVH, dnvalueVV],axis=1)
        dn_df = pd.DataFrame(data=dnvalues, columns=['F' + str(x) for x in range(1, 2 * (exp_bands) + 1)])
        dn_df['label'] = l
    
        dnvalues = np.concatenate([dnvalueVH, dnvalueVV],axis=1)
        dn_df = pd.DataFrame(data=dnvalues, columns=['F' + str(x) for x in range(1, 2 * (exp_bands) + 1)])
        dn_df['label'] = l
    
        return dn_df
            
    except Exception as E:    
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
'''        

def latlong2DN(label_file, processing_dir, period, trdata_dir):
    try:
        in_fileVH_List = glob('{}/Timeseries_**_VH.tif'.format(processing_dir))
        print("in_fileVH : {}".format(in_fileVH_List))
        in_fileVH = ''
        for img_path1 in in_fileVH_List:
          in_fileVH = img_path1 
          print("in_fileVH : {}".format(in_fileVH))
        
        in_fileVV_List = glob('{}/Timeseries_**_VV.tif'.format(processing_dir))
        print("in_fileVV : {}".format(in_fileVV_List))
        in_fileVV = ''
        for img_path2 in in_fileVV_List:
          in_fileVV = img_path2 
          print("in_fileVV : {}".format(in_fileVV))
                
        image_VH, _ = load_bands(in_fileVH)
        print("image_VH : {}".format(image_VH))
        image_VV, _ = load_bands(in_fileVV)
        print("image_VV : {}".format(image_VV))

        row, col = latlong2rowcol(in_fileVH, label_file['X'], label_file['Y'])
        print("row : {} and col : {}".format(row, col))
        dnvalueVH = image_VH[:, row, col].transpose()
        print("dnvalueVH  : {}".format(dnvalueVH))
        dnvalueVV = image_VV[:, row, col].transpose()
        print("dnvalueVH : {}".format(dnvalueVH))

        # if the no. of. bands is less than period*3, add zero values
        exp_bands = math.ceil((period*30)/12)
        print("exp_bands : {}".format(exp_bands))
        if dnvalueVH.shape[-1] < exp_bands:
            dnvalueVH = np.append(dnvalueVH, np.zeros((dnvalueVH.shape[0], exp_bands - dnvalueVH.shape[1])), axis=1)
            dnvalueVV = np.append(dnvalueVV, np.zeros((dnvalueVV.shape[0], exp_bands - dnvalueVV.shape[1])), axis=1)
        elif dnvalueVH.shape[-1] > exp_bands:
            dnvalueVH = dnvalueVH[:, exp_bands]
            dnvalueVV = dnvalueVV[:, exp_bands]

        dnvalues = np.concatenate([dnvalueVH, dnvalueVV],axis=1)
        dn_df = pd.DataFrame(data=dnvalues, columns=['F' + str(x) for x in range(1, 2 * (exp_bands) + 1)])
        dn_df['label'] = label_file['label_id']

        train_csv_path = os.path.join(trdata_dir, 'training_data_DNwlabels.csv')
        print("train_csv_path : {}".format(train_csv_path))
        dn_df.to_csv(train_csv_path, index=False)
        return train_csv_path

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))


def validate(model, dataset):
    try:
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batchX, batchy in dataset.get_next_batch(use_data='validation', batch_size=10):
                ypred = model(batchX)
                yp = ypred.argmax(axis=1).view(-1).numpy().tolist()
                yt = batchy.view(-1).numpy().tolist()
                y_pred.extend(yp)
                y_true.extend(yt)
        return get_metrics(ytrue=y_true, ypred=y_pred)

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))


def train(train_fname, num_epochs, model_file):
    try:
        dataset = CropDataset(train_fname)
        model = CropClassifier()
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        best_f1 = 0
        best_metrics = []
        for e in range(num_epochs):
            losses = []
            epoch_num = e + 1
            for batchX, batchy in dataset.get_next_batch(batch_size=100):
                optimizer.zero_grad()
                ypred = model(batchX)
                loss = loss_function(ypred, batchy.view(-1))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            acc, bacc, pre, rec, f1s, cfm = validate(model, dataset)
            if f1s > best_f1:
                best_f1 = f1s
                best_metrics = [acc, bacc, pre, rec, f1s, cfm]
                torch.save(model.state_dict(), model_file)
            loggerupds.info(f'EPOCH {epoch_num} :: Train Loss: {sum(losses)/len(losses):.6f} :: Validation Metrics: A[{acc:.2f}], B[{bacc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')
            print(
                f'EPOCH {epoch_num} :: Train Loss: {sum(losses)/len(losses):.6f} :: Validation Metrics: A[{acc:.2f}], B[{bacc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')

        print(f'==========')
        print(f'BEST MODEL')
        print(model_file)
        print(
            f'Validation Metrics: A[{best_metrics[0]:.2f}], B[{best_metrics[1]:.2f}], P[{best_metrics[2]:.2f}], R[{best_metrics[3]:.2f}], F[{best_metrics[4]:.2f}]')

        loggerupds.info(f'==========')
        loggerupds.info(f'BEST MODEL')
        #loggerupds.info(f'Validation Metrics: A[{best_metrics[0]:.2f}], P[{best_metrics[1]:.2f}], R[{best_metrics[2]:.2f}], F[{best_metrics[3]:.2f}]')
        loggerupds.info(f'Validation Metrics: A[{best_metrics[0]:.2f}], B[{best_metrics[1]:.2f}], P[{best_metrics[2]:.2f}], R[{best_metrics[3]:.2f}], F[{best_metrics[4]:.2f}]')

        y_pred = []
        y_true = []
        with torch.no_grad():
            for batchX, batchy in dataset.get_next_batch(use_data='test'):
                #X_test_normalization = pd.DataFrame(scaler.transform(batchX),columns = batchX.columns)
                #test_data_normalized = scaler.transform(batchX.reshape(-1, 1))
                #test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)
                #ypred = model(test_data_normalized)
                #print("type of batchX : {} and type of batchy : {}".format(type(batchX), type(batchy)))
                #print("batchX : {} and batchy : {}".format(batchX, batchy))
                ypred = model(batchX)
                yp = ypred.argmax(axis=1).view(-1).numpy().tolist()
                yt = batchy.view(-1).numpy().tolist()
                y_pred.extend(yp)
                y_true.extend(yt)
        acc, bacc, pre, rec, f1s, cfm = get_metrics(ypred=y_pred, ytrue=y_true)
        #print(f'Test Metrics: A[{acc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')
        print(f'Test Metrics: A[{acc:.2f}], B[{bacc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')
        #loggerupds.info(f'Test Metrics: A[{acc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')
        loggerupds.info(f'Test Metrics: A[{acc:.2f}], B[{bacc:.2f}], P[{pre:.2f}], R[{rec:.2f}], F[{f1s:.2f}]')
        # plot confusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.show()

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))



def launch_model_training(user_id, job_id, start_date, period, label_file_path, clat, clong, clabel, crop_name):
    try:
        loggerupds = logging.getLogger('update')
        loggerupds.info('INFO: Inside model_training.py')
        # call_update_modaltrainning(args.job_id,'Processing','2')
        # get project working directory
        project_dir = configure_procdirec(user_id, job_id)
        trdata_dir = os.path.join(project_dir, 'training_data')
        os.makedirs(trdata_dir, exist_ok=True)

        # create a bounding box json from the point coordinates
        # convert excel into shapefile
        pt_gjson = excel_to_geojson(label_file_path, clat, clong, clabel)
        ptshp_path = os.path.join(trdata_dir, "TrainingData_Point.shp")
        save_shapefile(pt_gjson, fname=ptshp_path, geometry="Point")

        # create a convex hull polygon shapefile from the points shapefile
        polyshp_path = os.path.join(trdata_dir, "TraningData_BBoxPoly.shp")
        pt2polygon_convexhull(ptshp_path, polyshp_path)

        # convert polygon shp to polygon.geojson
        aoi_path = os.path.join(trdata_dir, 'TrainingData_BBox.geojson')
        geodf = geopandas.read_file(polyshp_path)
        geodf.to_file(aoi_path, driver='GeoJSON')

        # calculate end date using period
        end_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=30 * period)
        end_date = end_date.strftime("%Y-%m-%d")
        # download the dataset required for model training
        loggerupds.info('INFO: Calling Data Preparation code')
        output_timeseries_path, output_pth, _ = launch_dataprep(
            user_id=user_id,
            job_id=job_id,
            start_date=start_date,
            end_date=end_date,
            aoi_path=aoi_path,
            product_type='GRD',
            direction='DESCENDING_VVVH',
            ls_mask=False,
            rmv_speckle=False,
            resolution=10
        )
        
        # create a csv containing the VV and VH DN values and label info
        # define the processing directory and the VH, VV files to be used for model prediction
        #processing_dir = os.path.dirname(os.path.dirname(output_pth[0]))
        processing_dir = output_timeseries_path
        print("processing_dir : {}".format(processing_dir))
        # read the excel file and rename lat long columns
        loggerupds.info('INFO: Reading the excel file containing training labels')
        label_file = read_label_file(label_file_path, clabel, crop_name)
        print("label_file : {}".format(label_file))
        
        '''
        loggerupds.info('INFO: Writing the training labels along with DN values to a csv file')
        # Get pixel values of the samples and save to csv
        dn_final = pd.DataFrame()
        before_string_fitter = '{0}/{1}_*_VH*.tif'.format(output_timeseries_path, user_id)
        print("before_string_fitter : {}".format(before_string_fitter))
        after_string_fitter = before_string_fitter.replace("//","/")
        print("after_string_fitter : {}".format(after_string_fitter))
        #files = glb(after_string_fitter)
        files = glob(after_string_fitter)
        print("glob - files list : {}".format(files))
        
        # define csv file name training_data_DNwlabels.csv
        train_csv_path = os.path.join(output_timeseries_path, 'training_data_DNwlabels.csv')
        print("train_csv_path : {}".format(train_csv_path))
        
        loggerupds.info('INFO: Writing the training labels along with DN values to a csv file')
        
        #for img_path in glob('{}/Sentinel1_VH*.tif'.format(data_dir)):
        dn_final = pd.DataFrame()
        for img_path in files:
            print("img_path : {}".format(img_path))
            in_fileVH = img_path
            print("in_fileVH : {}".format(in_fileVH))
            in_fileVV = '{}/{}'.format(output_timeseries_path, os.path.basename(img_path).replace('VH', 'VV'))
            print("in_fileVH : {}".format(in_fileVH))
    
            dn_df = latlong2DN(label_file, in_fileVH, in_fileVV, period)
            print("dn_df : {}".format(dn_df))
            print("dn_df.shape : {}".format(dn_df.shape))
            dn_final = dn_final.append(dn_df, ignore_index=True)
            #dn_final.to_csv('{}/{}.csv'.format(data_dir, csvname), index=False)    
            dn_final.to_csv(train_csv_path, index=False)
        '''
        loggerupds.info('INFO: Writing the training labels along with DN values to a csv file')
        train_csv_path = latlong2DN(label_file, processing_dir, period, trdata_dir)
        print("train_csv_path : {}".format(train_csv_path))

        # declare model parameters
        num_epochs = 200
        print("num_epochs : {}".format(num_epochs))

        model_path = os.path.join(project_dir, 'trained_model.pth')
        print("model_path : {}".format(model_path))
        # call_update_modaltrainning(args.job_id,model_path,'3')
        # start model training
        loggerupds.info('INFO: Start model training')

        train(train_fname=train_csv_path, num_epochs=num_epochs, model_file=model_path)
        loggerupds.info('INFO: Exit model training code')
        
        print("saved model path : {}".format(model_path))
        return model_path

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        
        # call_update_modaltrainning(args.job_id,'Error:Lauch Modal trainning','9')
        
        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
        
def launch_model_training_test(user_id, job_id, start_date, period, label_file_path, clat, clong, clabel, crop_name):
    try:
        loggerupds = logging.getLogger('update')
        loggerupds.info('INFO: Inside model_training.py')
        # call_update_modaltrainning(args.job_id,'Processing','2')
        # get project working directory
        project_dir = configure_procdirec(user_id, job_id)
        trdata_dir = os.path.join(project_dir, 'training_data')
        os.makedirs(trdata_dir, exist_ok=True)

        # create a bounding box json from the point coordinates
        # convert excel into shapefile
        pt_gjson = excel_to_geojson(label_file_path, clat, clong, clabel)
        ptshp_path = os.path.join(trdata_dir, "TrainingData_Point.shp")
        save_shapefile(pt_gjson, fname=ptshp_path, geometry="Point")

        # create a convex hull polygon shapefile from the points shapefile
        polyshp_path = os.path.join(trdata_dir, "TraningData_BBoxPoly.shp")
        pt2polygon_convexhull(ptshp_path, polyshp_path)

        # convert polygon shp to polygon.geojson
        aoi_path = os.path.join(trdata_dir, 'TrainingData_BBox.geojson')
        print("launch_model_training_test - aoi_path : {}".format(aoi_path))
        geodf = geopandas.read_file(polyshp_path)
        geodf.to_file(aoi_path, driver='GeoJSON')

        # calculate end date using period
        end_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=30 * period)
        end_date = end_date.strftime("%Y-%m-%d")
        # download the dataset required for model training
        loggerupds.info('INFO: Calling Data Preparation code')
        '''
        output_timeseries_path, output_pth, _ = launch_dataprep(
            user_id=user_id,
            job_id=job_id,
            start_date=start_date,
            end_date=end_date,
            aoi_path=aoi_path,
            product_type='GRD',
            direction='DESCENDING_VVVH',
            ls_mask=False,
            rmv_speckle=False,
            resolution=10
        )
        '''
        # create a csv containing the VV and VH DN values and label info
        # define the processing directory and the VH, VV files to be used for model prediction
        #processing_dir = os.path.dirname(os.path.dirname(output_pth[0]))
        #processing_dir = output_timeseries_path
        #processing_dir = '/home/ubuntu/mys3bucket/AgriX_Data/64/644447_test/Timeseries' 
        processing_dir = '/home/ubuntu/mys3bucket/AgriX_Data/64/644444_test/Timeseries' 
        #processing_dir = '/home/ubuntu/mys3bucket/AgriX_Data/64/640786_test/Timeseries'
        print("processing_dir : {}".format(processing_dir))
        # read the excel file and rename lat long columns
        loggerupds.info('INFO: Reading the excel file containing training labels')
        label_file = read_label_file(label_file_path, clabel, crop_name)
        print("label_file : {}".format(label_file))
        
        loggerupds.info('INFO: Writing the training labels along with DN values to a csv file')
        train_csv_path = latlong2DN(label_file, processing_dir, period, trdata_dir)
        print("train_csv_path : {}".format(train_csv_path))

        # declare model parameters
        num_epochs = 200
        print("num_epochs : {}".format(num_epochs))

        model_path = os.path.join(project_dir, 'trained_model.pth')
        print("model_path : {}".format(model_path))
        # call_update_modaltrainning(args.job_id,model_path,'3')
        # start model training
        loggerupds.info('INFO: Start model training')

        train(train_fname=train_csv_path, num_epochs=num_epochs, model_file=model_path)
        loggerupds.info('INFO: Exit model training code')        
        print("saved model path : {}".format(model_path))
        
        # call_update_modaltrainning on database connnector
        #call_update_modaltrainning(args.job_id,model_path,'3')
        
        # return model path
        return model_path

    except Exception as E:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        
        # call_update_modaltrainning(args.job_id,'Error:Lauch Modal trainning','9')
        #call_update_modaltrainning(args.job_id,'Error:Lauch Modal trainning','9')
        
        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))


if __name__ == "__main__":
    '''
    user_id = '64'
    job_id = '644447_test'
    start_date = '2020-08-21' 
    period = 3
    label_file_path = '/home/ubuntu/Amuthan_weed_classification_venv/codes/ModalTrainning.xlsx'
    clat = 'Latitude'
    clong = 'Longitude'
    clabel = 'Crop Name'
    crop_name = 'Paddy Ii'
    '''
    '''
    user_id = '64'
    job_id = '640118_test'
    start_date = '2020-08-21' 
    period = 3
    label_file_path = '/home/ubuntu/Amuthan_weed_classification_venv/codes/ModalTrainning.xlsx'
    clat = 'Latitude'
    clong = 'Longitude'
    clabel = 'Crop Name'
    crop_name = 'Paddy Ii'
    '''
    
    user_id = '64'
    job_id = '640786_test'
    start_date = '2020-08-21' 
    period = 3
    label_file_path = '/home/ubuntu/Amuthan_weed_classification_venv/codes/ModalTrainning.xlsx'
    clat = 'Latitude'
    clong = 'Longitude'
    clabel = 'Crop Name'
    crop_name = 'Paddy Ii'
    
    '''
    user_id = '64'
    job_id = '644444_test'
    start_date = '2020-08-21' 
    period = 3
    label_file_path = '/home/ubuntu/Amuthan_weed_classification_venv/codes/ModalTrainning.xlsx'
    clat = 'Latitude'
    clong = 'Longitude'
    clabel = 'Crop Name'
    crop_name = 'Paddy Ii'
    '''
    '''
    user_id = '1'
    job_id = 'T12443'
    start_date = '2021-10-05' 
    period = 7
    label_file_path = '/home/ubuntu/Amuthan_weed_classification_venv/codes/ModalTrainning.xlsx'
    clat = 'Latitude'
    clong = 'Longitude'
    clabel = 'Crop Name'
    crop_name = 'Paddy Ii'
    '''
    model_path = launch_model_training_test(user_id, job_id, start_date, period, label_file_path, clat, clong, clabel, crop_name)
    print("model_path : {}".format(model_path))