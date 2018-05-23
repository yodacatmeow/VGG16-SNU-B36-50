"""
- Place this file inside of a folder (e.g. "001")
- change "class_id" in the code below to the folder name (in this case '1')
- run this code
- * if audio format of a file is ".m4a", then you need to change ".m4a" in the code below to "your format"
"""

import os

init = 0
class_id = '003'
class_id = class_id.zfill(3)    # zero pad the string "class_id"

for filename in os.listdir("."):
    if filename.find(".m4a") == -1:
        pass
    else:
        os.rename(filename, class_id + str(f'{init:03}') + ".m4a") #f'{init:03}' <-zeropadding (max 3digits)
        init = init + 1
