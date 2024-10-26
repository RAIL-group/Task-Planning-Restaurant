import numpy as np

PROCTHOR_DATA_PATH = "./data/procthor-data/data.jsonl"
PROCTHOR_DEF_METADATA = 'metadata'

# House dictionary keys in ProcTHOR used in this api
ROOMS_KEY = 'rooms'
DOORS_KEY = 'doors'
WALLS_KEY = 'walls'
OBJECTS_KEY = 'objects'

# Wall keys
PROC_WALL_POLYGON = 'polygon'

# Room Dictionary Keys in ProcTHOR used in this api
PROC_ROOM_FLOOR_POLYGON = 'floorPolygon'
PROC_ROOM_TYPE = 'roomType'
PROC_ROOM_ID = 'id'
PROC_SINGLE_DOOR_OFFSET = 0.2
PROC_DOUBLE_DOOR_OFFSET = 0.4

# Door Dictionary Keys in ProcTHOR used in this api
PROC_DOOR_ASSET_ID = 'assetId'  # Unique id (name with some other charecters(numeric or special))
PROC_DOOR_ID = 'id'  # Generic name with the info of the shared rooms 
PROC_DOOR_OPENABLE = 'openable'  # If the can be opened or not?
PROC_DOOR_ROOM_0_ID = 'room0'  # room0 and room1 are the rooms that can be acessed through this door
PROC_DOOR_ROOM_1_ID = 'room1'  # 2nd room
PROC_DOOR_WALL_0_ID = 'wall0'  # wall0 and wall1 are the walls of the corresponding rooms where the door is locatied
PROC_DOOR_WALL_1_ID = 'wall1'  # 2nd wall (at room1)
PROC_DOOR_ASSET_POSITION = 'assetPosition'  # new attribute that shows the position with respect to something(!)
PROC_DOOR_HOLE_POLYGON = 'holePolygon'  # probably the door hole in the wall [3d]

# Objects Dictionary Keys in ProcTHOR used in this api as CONTAINERS
PROC_CONTAINER_ASSET_ID = 'assetId'  # Unique id (name with some other charecters(numeric or special))
PROC_CONTAINER_ID = 'id'  # Generic asset name with some other properties
PROC_CONTAINER_IS_DIRTY = 'isDirty'  # Whether this container is dirty or not
PROC_CONTAINER_POSITION = 'position'
PROC_UNIQUE_CONTAINER_ID = 'ucid'
PROC_CONTAINER_NUM_CHILDREN = 'chnum'

# Children list in each object object (weird) are the OBJECTS in our map
PROC_OBJECTS_LIST = 'children'
PROC_OBJECT_ASSET_ID = 'assetId'  # Unique id (name with some other charecters(numeric or special))
PROC_OBJECT_ID = 'id'  # Generic asset name with some other properties
PROC_OBJECT_POSITION = 'position'
PROC_UNIQUE_OBJECT_ID = 'uoid'

# Agent keys
PROCTHOR_DEF_ROBOT = 'agent'
PROC_ROBOT_POSITION = 'position'

# 2D coordinates according to procthor
X_VAL = 'x'
Y_VAL = 'y'
Z_VAL = 'z'

IGNORE_ASSETS = ['BaseballBat', 'BasketBall', 'Boots', 'DeskLamp',
                 'TennisRacket', 'Faucet', 'HousePlant', 'Dumbbell',
                 'FloorLamp', 'HousePlant', 'RoomDecor', 'ShowerCurtain',
                 'ShowerHead', 'Television', 'VacuumCleaner', 'Photo',
                 'Painting']  # DogBed

def get_object_location(object_pos, mod_locations):
    for reg, reg_pos in mod_locations.items():
        if np.array_equal(object_pos, reg_pos):
            return reg
