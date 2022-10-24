import re
import json

data = []
with open("archive/Sarcasm_Headlines_Dataset_with_article_text.json") as f:
    json_str = ""
    i = 0
    for idx, l in enumerate(f.readlines()):
        i+=1
        if i<6:
            json_str += l
        if i == 6:
            json_str += "}"
            el = json.loads(json_str)
            data.append(el)
            json_str = ""
            i=0
print(len(data))