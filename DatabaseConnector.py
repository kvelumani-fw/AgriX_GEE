import sys
#import mysql.connector as sql
import mysql.connector

def call_update_data_and_product(jobId,VVFilepath,VHFilepath,Status,Band):
 try:      
        #conn = sql.connect( host="tnau-dev.c5uedrzo7co7.ap-south-1.rds.amazonaws.com", user="admin", password="AllIsWell!", database="agrex_prod", use_pure=True)
        conn = mysql.connector.connect( host="tnau-dev.c5uedrzo7co7.ap-south-1.rds.amazonaws.com", user="admin", password="AllIsWell!", database="agrex_prod", use_pure=True)
        cursor = conn.cursor()
        args = (jobId, VVFilepath,VHFilepath,Status,Band,'0')
        print("Job ID: "+jobId+ " VVFilepath: "+VVFilepath +" VHFilepath: "+VHFilepath + " Status: "+Status +" Bands: "+Band)
        result_args = cursor.callproc('PROC_UPDATE_DATAPRODUCT', args)
        conn.commit()
        print("Updated Succesfully")
 except mysql.connector.Error as error:
    print("Failed to execute stored procedure: {}".format(error))

 finally:
        cursor.close()
        conn.close()

def call_update_modaltrainning(jobId,outputpath,Status):
 try:      
        #conn = sql.connect( host="tnau-dev.c5uedrzo7co7.ap-south-1.rds.amazonaws.com", user="admin", password="AllIsWell!", database="agrex_prod", use_pure=True)
        conn = mysql.connector.connect( host="tnau-dev.c5uedrzo7co7.ap-south-1.rds.amazonaws.com", user="admin", password="AllIsWell!", database="agrex_prod", use_pure=True)
        cursor = conn.cursor()
        args = (jobId,outputpath,Status)
        print("Job ID: "+jobId+ "Filepath: "+outputpath +" Status: "+Status)
        result_args = cursor.callproc('PROC_UPDATE_MODALTRAINNING', args)
        conn.commit()
        print("Updated Succesfully")
 except mysql.connector.Error as error:
    print("Failed to execute stored procedure: {}".format(error))

 finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    #call_update_data_and_product('1001124','Failed','Failed','9','0')
    #call_update_data_and_product('22225','/home/ubuntu/mys3bucket/AgriX_Data/1/22225/Products/Rice_map.tif','/home/ubuntu/mys3bucket/AgriX_Data/1/22225/Products/Rice_map.tif','3','1')
    #call_update_data_and_product('12125','/home/ubuntu/mys3bucket/AgriX_Data/1/12125/Timeseries/Products/rice_map.tif','/home/ubuntu/mys3bucket/AgriX_Data/1/12125/Timeseries/Products/rice_map.tif','3','1')
    # update model trainning on database connnector
    call_update_modaltrainning('T12504', '/home/ubuntu/mys3bucket/AgriX_Data/1/T12504/trained_model.pth', '3')