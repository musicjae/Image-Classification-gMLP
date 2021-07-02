import werkzeug
from utils import *
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Flask, jsonify
from model import *
from flask_restplus import Api, Resource
from werkzeug.datastructures import FileStorage
from flask_cors import CORS
import numpy as np
import random
from PIL import Image
import io

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
    model = MLP.load_from_checkpoint('lightning_logs/trained-epoch=19-val_loss=0.00-basic-mlp-v4.ckpt',
                                     map_location=device,
                                     lr=args.lr,
                                     image_size=args.img_size)


####################################### Back-end #######################################

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
CORS(app)
api = Api(app, version='1.0', title='4 종류 강아지 이미지 분류 모델', description='이미지를 입력하시면 그 이미지 내 강아지의 종을 출력해줍니다.')
ns = api.namespace('dogclf')
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'

upload_parser = ns.parser()
upload_parser.add_argument('file',
                           location='files',
                           type=FileStorage)


@api.route('/dogclf/predict')
@api.expect(upload_parser)
class UploadDemo(Resource):
    def post(self):
        args = upload_parser.parse_args()
        img_bytes = args.get('file').read()
        imgbytes2image = Image.open(io.BytesIO(img_bytes))
        input_image = inference_transforms(imgbytes2image).unsqueeze(0)

        model.eval()
        output = model(input_image)

        index = output.data.numpy().argmax()

        return jsonify({'dog': str(decoder[index])})


if __name__ == '__main__':

    app.run(port=5000, host='0.0.0.0')
