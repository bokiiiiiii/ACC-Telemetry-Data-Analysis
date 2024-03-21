import time, keyboard
from typing import Any
import matplotlib.pyplot as plt

from ACCTelemetry import ACCTelemetry
from RealTimePlot import RealTimePlot




if __name__ == '__main__':
    
    ACCTelemetryData = ACCTelemetry()
    ACCTelemetryData.start()
    
    ACCRealTimePlot = RealTimePlot()

    
    while True:                      
            
        # Get the current ACC telemetry data 
        Current_ACCData: dict[str, Any] = ACCTelemetryData.getACCData()
        
        # Real-time plot current ACC data     
        ACCRealTimePlot.plot(Current_ACCData)
        


        time.sleep(0.01) # Pause delta time between loops
        
        
        # Press 'Esc' to stop the program 
        if keyboard.is_pressed('esc'):
            break    
    
   
               
    ACCTelemetryData.stop()
    
    plt.show()
