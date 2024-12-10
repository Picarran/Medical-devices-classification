from ultralytics.models import YOLO
import os
import ssl
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# 禁用SSL验证
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    print(ssl.get_default_verify_paths())
    model = YOLO(model='./run/train/exp3/weights/best.pt')
    model.val(data='./data.yaml', split='val', batch=1, device='0', project='run/val', name='exp',
              half=False, )