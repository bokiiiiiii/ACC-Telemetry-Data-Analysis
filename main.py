import time, keyboard
from ACCTelemetry import ACCTelemetry
from RealTimePlot import RealTimePlot

import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    ACCTelemetryData = ACCTelemetry()
    ACCTelemetryData.start()
    
    ACCRealTimePlot = RealTimePlot()

    
    while True:
            
        # Current_ACCphysicsData = ACCTelemetryData.getphysicsData()
        # Current_ACCgraphicData = ACCTelemetryData.getgraphicData()
        # Current_ACCstaticData = ACCTelemetryData.getstaticData()
        # Current_ACCData = {**Current_ACCphysicsData, **Current_ACCgraphicData, **Current_ACCstaticData}               
        
        Current_ACCData = ACCTelemetryData.getACCData()
        # Real-time plot current ACC data     
        ACCRealTimePlot.plot(Current_ACCData)
        

        time.sleep(0.01) # Pause delta time between loops
        
        
        # Press 'Esc' to stop the program 
        if keyboard.is_pressed('esc'):
            break    
    
   
               
    ACCTelemetryData.stop()
    
    plt.show()
