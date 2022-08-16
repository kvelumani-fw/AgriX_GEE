import gevent
import os
os.environ["GEVENT_SUPPORT"] = 'True'

from signalr import Connection
from requests import Session

def Update_Filepath(job_id,stacked_path):
    """

    :param job_id:
    :param stacked_path:
    :return:
    """
    with Session() as session:
     print("Connecting...")
     #create a connection
     connection = Connection("http://ec2-3-108-57-241.ap-south-1.compute.amazonaws.com:9888/signalr/hubs/",session)
    
    
     #get chat hub
     chat = connection.register_hub('NotifyHub')

     #start a connection
     connection.start()
     print("Connected")
     #start connection, optionally can be connection.start()
       
     chat.server.invoke('UpdateDatabse', job_id, stacked_path)
     
     #connection.close() 
     print("Send succesfully")

    return "Success"