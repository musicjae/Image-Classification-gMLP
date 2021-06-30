import werkzeug
import pytorch_lightning as pl
from utils import *
werkzeug.cached_property = werkzeug.utils.cached_property
import os
from flask import Flask, request, render_template, jsonify
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from model import *
from flask_restplus import Api, Resource
from werkzeug.datastructures import FileStorage
import py_eureka_client.eureka_client as eureka_client
import py_eureka_client.netint_utils as netint_utils
import argparse, configparser
from flask_cors import CORS
import numpy as np
import random
from PIL import Image

random_seed = 827

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else ('cpu'))

if args.mode == 'gmlp':
    model = gMLPForImageClassification.load_from_checkpoint('lightning_logs/trained-epoch=19-val_loss=0.00-gmlp.ckpt',
                                                            map_location=device,
                                                             lr=args.lr,
                                                             image_size=args.img_size
                                                             ) # Trained Model load
elif args.mode == 'basic-mlp':
    model = MLP.load_from_checkpoint('lightning_logs/trained-epoch=19-val_loss=0.00-basic-mlp.ckpt',
                                     map_location=device,
                                     lr=args.lr,
                                     image_size=args.img_size)



model.eval()
print(model)

testsp = Image.open('sample/img/000000000023.jpg')
print('---------------original--------------',testsp.size)
testst = inference_transforms(testsp).reshape((256,3,16,16))
print('---------------processed--------------',testst.size())
print('---------------model--------------',model(testst))

# app = Flask(__name__)
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# CORS(app)
# api = Api(app, version='1.0', title='딥러닝 기반 이미지를 이용한 작명 서비스', description='이미지를 입력하면 이미지 내 동물과 어울리는 이름을 추천해줍니다.')
# ns = api.namespace('namerec')
# app.config.SWAGGER_UI_DOC_EXPANSION = 'full'
#
# upload_parser = ns.parser()
# upload_parser.add_argument('file',
#                            location='files',
#                            type=FileStorage)
#
#
# @api.route('/namerec/predict')
# @api.expect(upload_parser)
# class UploadDemo(Resource):
#     def post(self):
#         args = upload_parser.parse_args()
#         img_bytes = args.get('file').read()
#         output = infer(img_bytes)
#         output = model(output)
#         index = output.data.numpy().argmax()
#
#         return jsonify({'dog': str(encoder[index])})
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='which server?')
#     # parser.add_argument('--server_name', '-s', type=str, default='test', help='service, test, prod')
#     parser.add_argument('--port_number', '-p', type=int, default=30021, help='port number default id 30015')
#     args = parser.parse_args()
#     #
#     # config = configparser.ConfigParser()
#     #
#     # config.read('../server_data.ini')
#     #
#     # ip = config[args.server_name]['ip']
#     port_number = args.port_number
#     #
#     # if ip == 'None':
#     #     ip = netint_utils.get_first_non_loopback_ip()
#     #
#     # eureka_host = os.environ.get('EUREKA_HOST', config[args.server_name]['server'])
#     # server_host = os.environ.get('SERVER_HOST', ip)
#     #
#     # eureka_client.init(eureka_server=f"{eureka_host}:38761/eureka/",
#     #                    app_name="dog-classification",
#     #                    instance_ip=server_host,
#     #                    instance_host=server_host,
#     #                    instance_port=port_number)
#
#     app.run(port=port_number, host='0.0.0.0')
