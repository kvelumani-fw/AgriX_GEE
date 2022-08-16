from geoserver.catalog import Catalog

def configure_geoserver_data(jobId,filepath,request_type_id):
    # Initialize the library
    cat = Catalog("http://ec2-13-234-203-246.ap-south-1.compute.amazonaws.com:8080/geoserver/rest",username="admin", password="geoserver")
    topp = cat.get_workspace("AgrixConfigure")
    # For creating workspace
    cat.create_coveragestore( jobId, workspace='AgrixConfigure', path=filepath, type='GeoTIFF',create_layer=True, layer_name=jobId,source_name=jobId, upload_data=True,contet_type="image/tiff",overwrite=True)  
    # For creating the style file for raster data dynamically and connect it with layer
    layer = cat.get_layer(jobId)
    if request_type_id==1:
        layer.default_style = 'MultiBandRasterStyle'
        print("layer.default_style : {}".format(layer.default_style))
    else:
        layer.default_style = 'SingleBandStyle'
        print("layer.default_style : {}".format(layer.default_style))
    cat.save(layer)  
    print("create coverage store")
    print("success")
    return 'success'

if __name__ == "__main__":
    '''
    jobId = '640000_GserverTest_13Oct'
    request_type_id = 1
    filepath = '/home/ubuntu/64/640000_test/processing/92/Timeseries_ANLD/Timeseries_ANLD_stacked_5_VH.tif'
    configure_geoserver_data(jobId, filepath, request_type_id)
    '''
    # Amuthan Test
    '''
    jobId = '316298_VH'
    request_type_id = 1
    filepath = '/home/ubuntu/mys3bucket/AgriX_Data/91/316298/Timeseries/Timeseries_stacked_VH.tif'
    configure_geoserver_data(jobId, filepath, request_type_id)
    '''
    
    jobId = '12125_VH'
    request_type_id = 2    
    filepath = '/home/ubuntu/mys3bucket/AgriX_Data/1/12125/Timeseries/Products/rice_map.tif'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = '12125_VV'
    request_type_id = 2    
    filepath = '/home/ubuntu/mys3bucket/AgriX_Data/1/12125/Timeseries/Products/rice_map.tif'
    configure_geoserver_data(jobId, filepath, request_type_id)
