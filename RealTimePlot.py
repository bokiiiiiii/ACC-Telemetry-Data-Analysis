import matplotlib.pyplot as plt

class RealTimePlot:
    
    def __init__(self):
        
        mem_x = []
        mem_y = []
        mem_z = []

        # Initialize the plot
        plt.style.use('dark_background')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
                
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.set_title("Real-Time Car Coordinate", weight = 'bold', size = 15)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(-1500, 1500)
        ax.set_ylim(-1500, 1500)
        ax.axis('equal')
        
        line, = ax.plot(mem_z, mem_x, label = 'Racing Line')
        line_color = line.get_color()
        marker, = ax.plot(mem_z, mem_x, '.', color = line_color, label = 'Current Position')
        
        ax.legend()
        
        self.mem_x = mem_x
        self.mem_y = mem_y
        self.mem_z = mem_z
        self.fig = fig
        self.ax = ax
        self.line = line
        self.line_color = line_color
        self.marker = marker
        
    
    def plot(self, Current_ACCData):
        
        mem_x = self.mem_x
        mem_y = self.mem_y
        mem_z = self.mem_z
        fig = self.fig
        ax = self.ax
        line = self.line
        line_color = self.line_color
        marker = self.marker
           
        
        Current_carCoordinates = Current_ACCData['carCoordinates']
        Current_iCurrentTime = Current_ACCData['iCurrentTime']
        Current_mycarCoordinates = Current_carCoordinates[:3]
        
        
        print(Current_mycarCoordinates)
        x = Current_mycarCoordinates[0] # x
        y = Current_mycarCoordinates[1] # y
        z = Current_mycarCoordinates[2] # z
        t = Current_iCurrentTime*0.001  # ms >> s


        mem_x.append(x)
        mem_y.append(y)
        mem_z.append(-z)
    
        
        line.set_data(mem_x, mem_z)
        marker.set_data(mem_x[-1], mem_z[-1])
        
        
        plt.draw()
        plt.pause(0.01)              