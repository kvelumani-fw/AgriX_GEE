import argparse
import logging
import os
import sys

from data_preparation import configure_procdirec
from model_training import launch_model_training
from DatabaseConnector import call_update_modaltrainning

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        # receive the arguments from the front end
        parser = argparse.ArgumentParser()
        parser.add_argument("--user_id",type=str, help="Unique User id")
        parser.add_argument("--job_id",type=str,help="unique job id assigned to the job request")
        parser.add_argument("--start_date",type=str, help="Start date for data download, format: yyyy-mm-dd")
        parser.add_argument("--period",type=int,default=3, help="The time period in months for model training - 3 or 6 months")
        parser.add_argument("--label_file_path",type=str, help="Path to the excel sheet containing the training data labels")
        parser.add_argument("--lat_long_label_cols",type=str, help="Names of the columns containing latitude,longitude and label data seperated by , Example='lat,long,labels'")
        parser.add_argument("--crop_name", type=str, help="Crop value as stored in the excel file ")
        args = parser.parse_args()
        # call_update_modaltrainning(args.job_id,'Loading...','2')
        clat, clong, clabel = args.lat_long_label_cols.split(',')


        project_dir = configure_procdirec(args.user_id, args.job_id)

        setup_logger('update', '{}/update.log'.format(project_dir))
        loggerupds = logging.getLogger('update')
        loggerupds.info('INFO: Column parameter received: {}'.format(args.lat_long_label_cols))
        loggerupds.info('INFO: 7th argument crop_name received: {}'.format(args.crop_name))

        loggerupds.info('INFO: Calling the model training funciton')

        # call script for model training
        model_path = launch_model_training(args.user_id, args.job_id, args.start_date, args.period,
                                           args.label_file_path, clat, clong, clabel, args.crop_name)

        # call_update_modaltrainning(args.job_id,model_path,'3')
    except Exception as E:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        # call_update_modaltrainning(args.job_id,'Error On Main Argument','9')
        loggerupds = logging.getLogger('update')
        loggerupds.error("Error: . {}-  {}: {}".format(fname, exc_tb.tb_lineno, E))
        print(E)


if __name__ == "__main__":
    main()





#