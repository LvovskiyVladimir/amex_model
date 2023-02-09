from flask import Flask
from flask_restx import Resource, Api, reqparse, fields

from model import ModelWrapper

MAX_MODEL_NUM = 10
MODELS_DICT = dict()

app = Flask(__name__)
app.config["BUNDLE_ERRORS"] = True
api = Api(app)


model_add = api.model(
    "Model.add.input", {
        "name":
        fields.String(
            required=True,
            title="Model name",
            description="Used as a key in local models storage; Must be unique;"
        ),
        "n_cv_fold":
        fields.Integer(required=True,
                      title="Number of cross-validation folds",
                      default='5'),
        "num_boost_round":
                fields.Integer(
                    required=True,
                    title="Number of boosting rounds",
                    default=10500),
        "boosting_type":
                fields.String(
                            required=True,
                            title="Boosting type",
                            default='dart',
                            description="gbdt or dart or rf"
                        )
    })

model_predict = api.model(
    "Model.predict.input", {
        "name":
        fields.String(required=True,
                      title="Model name",
                      description="Name of your existing trained model;"),
        "data_to_predict":
            fields.String(required=True,
                         title="Data to predict",
                         default="input_dir/",
                         description="path to data to predict"),
    })

parserRemove = reqparse.RequestParser(bundle_errors=True)
parserRemove.add_argument("name",
                          type=str,
                          required=True,
                          help="Name of a model you want to remove",
                          location="args")

parserTrain = reqparse.RequestParser(bundle_errors=True)
parserTrain.add_argument("name",
                         type=str,
                         required=True,
                         help="Name of a model you want to train",
                         location="args")

parserTrain.add_argument("dataset_path",
                         type=str,
                         required=True,
                         help="path to train dataset",
                         location="args")

parserPredict = reqparse.RequestParser(bundle_errors=True)
parserPredict.add_argument("name",
                         type=str,
                         required=True,
                         help="Name of a model you want to use",
                         location="args")

parserPredict.add_argument("dataset_path",
                         type=str,
                         required=True,
                         help="path to predict dataset",
                         location="args")


@api.route("/models/list")
class ModelList(Resource):
    @api.doc(responses={201: "Success"})
    def get(self):
        return {
            "models": {
                i: {
                    "type":
                    MODELS_DICT[i].type,
                    "is fitted":
                    MODELS_DICT[i].fitted,
                    "train_score":
                    None if not MODELS_DICT[i].fitted else
                    MODELS_DICT[i].train_score,
                    "test_score":
                    None if not MODELS_DICT[i].test_score else
                    MODELS_DICT[i].test_score,
                }
                for i in MODELS_DICT.keys()
            }
        }, 201


@api.route("/models/add")
class ModelAdd(Resource):
    @api.expect(model_add)
    @api.doc(
        responses={
            201: "Success",
            401: "'params' error; Params must be a valid json or dict",
            402:
            "Error while initializing model; See description for more info",
            403: "Model with a given name already exists",
            408: "The max number of models has been reached"
        })
    def post(self):
        __name = api.payload["name"]
        __n_cv_fold = api.payload["n_cv_fold"]
        __num_boost_round = api.payload["num_boost_round"]
        __boosting_type = api.payload["boosting_type"]

        if len(MODELS_DICT) >= MAX_MODEL_NUM:
            return {
                "status":
                "Failed",
                "message":
                "The max number of models has been reached; You must delete one before creating another"
            }, 408

        if __name not in MODELS_DICT.keys():
            try:
                MODELS_DICT[__name] = ModelWrapper(__name, __n_cv_fold, __num_boost_round, __boosting_type)
                return {"status": "OK", "message": "Model created!"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 402
        else:
            return {
                "status": "Failed",
                "message": "Model with a given name already exists"
            }, 403


@api.route("/models/remove")
class ModelRemove(Resource):
    @api.expect(parserRemove)
    @api.doc(responses={
        201: "Success",
        404: "Model with a given name does not exist"
    })
    def delete(self):
        __name = parserRemove.parse_args()["name"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist"
            }, 404
        else:
            MODELS_DICT.pop(__name)
            return {"status": "OK", "message": "Model removed!"}, 201


@api.route("/models/train")
class ModelTrain(Resource):
    @api.expect(parserTrain)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while training model; See description for more info"
        })
    def get(self):
        __name = parserTrain.parse_args()["name"]
        __dataset_path = parserTrain.parse_args()["dataset_path"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                MODELS_DICT[__name].fit(__dataset_path)
                return {"status": "OK", "message": f"Train score {MODELS_DICT[__name].train_score}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 406


@api.route("/models/predict")
class ModelPredict(Resource):
    @api.expect(parserPredict)
    @api.doc(
        responses={
            201: "Success",
            404: "Model with a given name does not exist",
            406: "Error while testing model; See description for more info"
        })
    def get(self):
        __name = parserTrain.parse_args()["name"]
        __dataset_path = parserTrain.parse_args()["dataset_path"]
        if __name not in MODELS_DICT.keys():
            return {
                "status": "Failed",
                "message": "Model with a given name does not exist!"
            }, 404
        else:
            try:
                preds_path = MODELS_DICT[__name].predict(__dataset_path)
                return {"status": "OK", "message": f"Results have been saved as {preds_path}"}, 201
            except Exception as e:
                return {
                    "status": "Failed",
                    "message": getattr(e, "message", repr(e))
                }, 406


if __name__ == "__main__":
    app.run(debug=True)
