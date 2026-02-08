import os.path
import json 

def load_key(keyname: str):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'keys.json'
    file_path = os.path.join(cur_dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            keys = json.load(file)
        if keyname in keys and keys[keyname]:
            return keys[keyname]
        else:
            return 'no such key or value'
    else:
        return 'cur path is not exist'

if __name__ == '__main__':
    print(load_key('LANGSMITH_API_KEY'))