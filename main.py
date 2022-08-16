from data_preparation import launch_dataprep,configure_procdirec
from product_preparation import launch_productprep
from DatabaseConnector import call_update_data_and_product
from GServer import configure_geoserver_data
import argparse
import logging
import sys
import os
# solving this error- ERROR 1: PROJ: proj_identify: /home/ubuntu/miniconda3/envs/AgriXGEE_Modified/share/proj/proj.db lacks DATABASE.LAYOUT.VERSION.MAJOR / DATABASE.LAYOUT.VERSION.MINOR metadata. It comes from another PROJ installation.
#import os
# Corresponding to oneself python Package installation address 
#os.environ['PROJ_LIB'] = '/home/ubuntu/miniconda3/envs/AgriXGEE_Modified/share/proj'
#os.environ['GDAL_DATA'] = '/home/ubuntu/miniconda3/envs/AgriXGEE_Modified/share'

def str2bool(v):
    if isinstance(v, bool):
       return 
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Received {}'.format(v))


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--user_id",type=str, help="Unique User id")
        parser.add_argument("--job_id",type=str,help="unique job id assigned to the job request")
        parser.add_argument("--start_date",type=str, help="Start date for data download, format: yyyy-mm-dd")
        parser.add_argument("--end_date",type=str, help="End date for data download, format: yyyy-mm-dd")
        parser.add_argument("--aoi_path",type=str, help="Path to area of interest stored as shapefile or geojson")
        parser.add_argument("--product_type",type=str, default='GRD', help="Product type, options=['SLC', 'GRD']")
        parser.add_argument("--product_crop_type",type=str, default=None, help="Crop type to generate product, options=['rice']")
        parser.add_argument("--request_type_id",type=int, default=1, help="Id to call data download/product launch, options=[1,2]")
        parser.add_argument("--train_period",type=int, default=14, help="Defined train period 14 - by defult it need to came from front end side calling")
        parser.add_argument("--model_path",type=str, default='/home/ubuntu/AgriX-Api/Crop_Analyser/models/rice_model.pth', help="Model used to generate the product")
        parser.add_argument("--direction",type=str, default='DESCENDING_VVVH', help="Desired direction, options=['DESCENDING_VVVH', 'ASCENDING_VVVH']") # to be checked
        parser.add_argument("--ls_mask",type=str2bool, default=False, help="If layover shadow mask should be applied") # to be checked
        parser.add_argument("--rmv_speckle",type=str2bool, default=False, help="If multi series speckle removal should be done") # to be checked
        args = parser.parse_args()
	
        project_dir = configure_procdirec(args.user_id, args.job_id)
        setup_logger('update', '{}/update.log'.format(project_dir))
        loggerupds = logging.getLogger('update')
        loggerupds.info("TEST: Configure and update file path and Image to DBS")

        if args.request_type_id == 1:
             call_update_data_and_product(args.job_id,'processing..','processing..','2','0')
             loggerupds.info('INFO: Calling Data Preparation code')
             output_timeseries_path, output_pth, band_count = launch_dataprep(
                user_id=args.user_id,
                job_id=args.job_id,
                start_date=args.start_date,
                end_date=args.end_date,
                aoi_path=args.aoi_path,
                product_type=args.product_type,
                direction=args.direction,
                ls_mask=args.ls_mask,
                rmv_speckle=args.rmv_speckle,
                resolution=10
             )
             print("output_pth list : {}".format(output_pth))
             loggerupds.info("INFO: Configure and update file path and Image to DBS")
             print("Configure and update file path and Image")
             call_update_data_and_product(args.job_id,output_pth[1],output_pth[0],'3',str(band_count))
             loggerupds.info("INFO: Configure and update file path and Images to Geoserver")
             configure_geoserver_data(args.job_id+'_VH',output_pth[0],args.request_type_id)
             configure_geoserver_data(args.job_id+'_VV',output_pth[1],args.request_type_id)
             loggerupds.info("INFO: Configured successfully")
             loggerupds.info(output_pth)
             print("Configured successfully")
             print(output_pth)
        else:
            call_update_data_and_product(args.job_id, 'processing..','processing..', '2','0')
            loggerupds.info('INFO: Calling Data Preparation code')
            output_timeseries_path, output_pth, band_count = launch_dataprep(
                user_id=args.user_id,
                job_id=args.job_id,
                start_date=args.start_date,
                end_date=args.end_date,
                aoi_path=args.aoi_path,
                product_type=args.product_type,
                direction=args.direction,
                ls_mask=False,
                rmv_speckle=False,
                resolution=10
            )
            loggerupds.info('Calling Product Preparation code')
            product_path, band_count = launch_productprep(
                crop_type=args.product_crop_type,
                model_path=args.model_path,
                data_dir=output_timeseries_path,
                aoi_path=args.aoi_path,
                job_id=args.job_id,
                user_id=args.user_id,
                train_period=args.train_period
            )
            print("Configure and update file path and Image")
            loggerupds.info("INFO: Finished product preparation, Path: {} , Band Count: {} ".format(product_path, band_count))
            loggerupds.info("INFO: Configure and update file path and Image to DBS")
            call_update_data_and_product(args.job_id, product_path[1],product_path[0], '3',str(band_count))
            loggerupds.info("INFO: Configure and update file path and Images to Geoserver")
            configure_geoserver_data(args.job_id+'_VH',product_path[0],args.request_type_id)
            configure_geoserver_data(args.job_id+'_VV',product_path[1],args.request_type_id)
            #configure_geoserver_data(args.job_id+'_VV',product_path[1])
            loggerupds.info("INFO: Configured successfully")
            print("INFO: Configured successfully")
            print(product_path)

    except Exception as E:
        #call_update_data_and_product(args.job_id,'Failed','Failed','9','0')

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
        print(E)


if __name__ == "__main__":
    main()
