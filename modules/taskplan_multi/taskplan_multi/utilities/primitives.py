import numpy as np

LAYOUT_DATA_PATH = "./data/procthor-data/data.jsonl"
ASSETS_FILE_NAME = "./data/procthor-data/data.jsonl"

# House dictionary keys in ProcTHOR used in this api
MAIN_DOMAIN = 'workshop'
ROOMS_KEY = 'rooms'
DOORS_KEY = 'doors'
OBJECTS_KEY = 'objects'
POLYGON = 'polygon'
ASSET_ID = 'assetId'  # Unique id (name with some other charecters(numeric or special))
ID = 'id'  # Generic name with the info 
DESCRIPTION = 'description'  # Generic name with the info of the shared rooms

ROOM_1 = 'tirestation'
ROOM_2 = 'mirrorstation'

BOT_1 = 'tire_bot'
BOT_2 = 'mirror_bot'

# 2D coordinates
X_VAL = 'x'
Y_VAL = 'y'
Z_VAL = 'z'