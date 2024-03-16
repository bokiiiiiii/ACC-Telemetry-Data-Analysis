import mmap, struct
from LocalParameters import LocalParametersClass

class ACCTelemetry(LocalParametersClass):
    
    def __init__(self):
        print('ACCData.init()')
        
        super().__init__()
        
        # Physics
        self.physics_shm_size = struct.calcsize(self.physics_layout)        
        self.mmapPhysics = None
        
        # Graphic
        self.graphic_shm_size = struct.calcsize(self.graphic_layout)        
        self.mmapGraphic = None
                
        # Static
        self.static_shm_size = struct.calcsize(self.static_layout)
        self.mmapStatic = None
        
        
    def start(self):
        print('ACCData.start()')
        
        if not self.mmapPhysics:
            self.mmapPhysics = mmap.mmap(-1, self.physics_shm_size, 'Local\\acpmf_physics',  access = mmap.ACCESS_READ)  
        if not self.mmapGraphic:    
            self.mmapGraphic = mmap.mmap(-1, self.graphic_shm_size, 'Local\\acpmf_graphics',  access = mmap.ACCESS_READ)
        if not self.mmapStatic:    
            self.mmapStatic = mmap.mmap(-1, self.static_shm_size, 'Local\\acpmf_static',  access = mmap.ACCESS_READ)
           

    def getphysicsData(self):
        self.mmapPhysics.seek(0)
        physics_rawData = self.mmapPhysics.read(self.physics_shm_size)
        physics_data = {}
        for index, value in enumerate(struct.unpack(self.physics_layout, physics_rawData)):
            physics_data[self.physics_fields[index]] = value
            
        self._convertphysicsData(physics_data)
        return physics_data


    def getgraphicData(self):
        self.mmapGraphic.seek(0)
        graphic_rawData = self.mmapGraphic.read(self.graphic_shm_size)
        graphic_data = {}
        for index, value in enumerate(struct.unpack(self.graphic_layout, graphic_rawData)):
            graphic_data[self.graphic_fields[index]] = value
            
        self._convertgraphicData(graphic_data)
        return graphic_data


    def getstaticData(self):
        self.mmapStatic.seek(0)
        static_rawData = self.mmapStatic.read(self.static_shm_size)
        static_data = {}
        for index, value in enumerate(struct.unpack(self.static_layout, static_rawData)):
            static_data[self.static_fields[index]] = value
            
        self._convertstaticData(static_data)
        return static_data


    def stop(self):
        print('ACCData.stop()')
        
        if self.mmapPhysics:
            self.mmapPhysics.close()
        self.mmapPhysics = None
        
        if self.mmapGraphic:
            self.mmapGraphic.close()
        self.mmapGraphic = None
        
        if self.mmapStatic:
            self.mmapStatic.close()
        self.mmapStatic = None


    def _convertphysicsData(self, data):
        # Make these conversions immediately while reading from shm
        # Combine X, Y, Z
        for newNameXYZ in self.physics_NameXYZList:
            data[newNameXYZ] = []
            for oldNameXYZ in [newNameXYZ + 'X', newNameXYZ + 'Y', newNameXYZ + 'Z']:
                data[newNameXYZ].append(data[oldNameXYZ])
                del data[oldNameXYZ]
        
        # Combine FL, FR, RL, RR            
        for newNameFLFRRLRR in self.physics_NameFLFRRLRRList:
            data[newNameFLFRRLRR] = []
            for oldNameFLFRRLRR in [newNameFLFRRLRR + 'FL', newNameFLFRRLRR + 'FR', newNameFLFRRLRR + 'RL', newNameFLFRRLRR + 'RR']:
                data[newNameFLFRRLRR].append(data[oldNameFLFRRLRR])
                del data[oldNameFLFRRLRR]
            

    def _convertgraphicData(self, data):
        # Make these conversions immediately while reading from shm
        # Combine x1, x2, ..., xn
        for newName in self.graphic_newNameList:
            data[newName] = []
            for oldName in getattr(self, f'{newName}_list'):
                data[newName].append(data[oldName])
                del data[oldName]

            # Decode bytes to char
            if newName in self.graphic_CharList:
                data_bytes = b"".join(data[newName])
                data_string = data_bytes.decode('utf-8', 'ignore')
                data[newName] = data_string
                


    def _convertstaticData(self, data):
        # Make these conversions immediately while reading from shm
        # Combine x1, x2, ..., xn
        for newName in self.static_newNameList:
            data[newName] = []
            for oldName in getattr(self, f'{newName}_list'):
                data[newName].append(data[oldName])
                del data[oldName]

            # Decode bytes to char
            data_bytes = b"".join(data[newName])
            data_string = data_bytes.decode('utf-8', 'ignore')
            data[newName] = data_string
        
        # Combine FL, FR, RL, RR            
        for newNameFLFRRLRR in self.static_NameFLFRRLRRList:
            data[newNameFLFRRLRR] = []
            for oldNameFLFRRLRR in [newNameFLFRRLRR + 'FL', newNameFLFRRLRR + 'FR', newNameFLFRRLRR + 'RL', newNameFLFRRLRR + 'RR']:
                data[newNameFLFRRLRR].append(data[oldNameFLFRRLRR])
                del data[oldNameFLFRRLRR]    
    
    
    def getACCData(self):
        Current_physicsData = self.getphysicsData()
        Current_graphicData = self.getgraphicData()
        Current_staticData = self.getstaticData()
        Current_Data = {**Current_physicsData, **Current_graphicData, **Current_staticData}
        return Current_Data          