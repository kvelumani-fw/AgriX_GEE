from geoserver.catalog import Catalog
import geoserver.util

def configure_geoserver_data(jobId,filepath,request_type_id):
    # Initialize the library
    cat = Catalog("http://ec2-13-234-203-246.ap-south-1.compute.amazonaws.com:8080/geoserver/rest",username="admin", password="geoserver")
    topp = cat.get_workspace("smartfarming")
    shapefile_plus_sidecars = geoserver.util.shapefile_and_friends(filepath)
    # For creating workspace
    cat.create_featurestore(jobId, shapefile_plus_sidecars, topp)
    # For creating the style file for raster data dynamically and connect it with layer
    layer = cat.get_layer(jobId)
    if request_type_id==1:
        layer.default_style = 'fmb_style'
        print("layer.default_style : {}".format(layer.default_style))
    else:
        layer.default_style = 'fmb_style'
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
    
    jobId = 'Tinnapalli_fmb_2_1'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step1/1/1_noisy'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_2'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/2/2_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_3'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/3/3_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_4'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/4/4_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_6'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/6/6_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_7'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/7/7_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_8'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/8/8_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_9'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/9/9_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_10'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/10/10_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    #10
    
    jobId = 'Tinnapalli_fmb_2_11'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/11/11_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    jobId = 'Tinnapalli_fmb_2_14'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/14/14_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    jobId = 'Tinnapalli_fmb_2_16'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/16/16_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
      
    jobId = 'Tinnapalli_fmb_2_20'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/20/20_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    #20#
    jobId = 'Tinnapalli_fmb_2_21'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/21/21_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_23'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/23/23_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_29'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/29/29_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    #30
    
    jobId = 'Tinnapalli_fmb_2_31'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/31/31_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    jobId = 'Tinnapalli_fmb_2_33'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/33/33_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    
    jobId = 'Tinnapalli_fmb_2_36'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/36/36_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    
    jobId = 'Tinnapalli_fmb_2_38'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/38/38_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_39'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/39/39_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    #40
    jobId = 'Tinnapalli_fmb_2_41'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/41/41_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_42'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/42/42_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_43'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/43/43_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_44'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/44/44_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    jobId = 'Tinnapalli_fmb_2_50'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/50/50_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    #50
    
    
    jobId = 'Tinnapalli_fmb_2_52'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/52/52_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_53'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/53/53_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_56'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/56/56_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_57'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/57/57_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_58'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/58/58_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    #60
    
    jobId = 'Tinnapalli_fmb_2_61'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/61/61_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_62'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/62/62_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_63'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/63/63_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_64'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/64/64_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_65'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/65/65_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_66'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/66/66_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_67'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/67/67_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_69'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/69/69_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_70'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/70/70_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    #70
    
    jobId = 'Tinnapalli_fmb_2_71'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/71/71_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_72'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/72/72_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_73'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/73/73_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_74'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/74/74_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_75'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/75/75_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_76'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/76/76_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    
    
    #80
    
    jobId = 'Tinnapalli_fmb_2_83'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/83/83_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_86'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/86/86_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_87'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/87/87_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_88'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/88/88_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_89'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/89/89_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    #90
    
    jobId = 'Tinnapalli_fmb_2_92'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/92/92_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_93'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/93/93_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_94'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/94/94_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_95'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/95/95_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_96'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/96/96_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
        
    jobId = 'Tinnapalli_fmb_2_98'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/98/98_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_99'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/99/99_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_100'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/100/100_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_101'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/101/101_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)
    
    jobId = 'Tinnapalli_fmb_2_103'
    request_type_id = 2    
    filepath = '/usr/share/geoserver/data_dir/workspaces/smartfarming/Tinnapalli_Fmb/FMBs_Step2/103/103_minordev'
    configure_geoserver_data(jobId, filepath, request_type_id)