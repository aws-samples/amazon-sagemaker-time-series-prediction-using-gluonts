import os
os.system('pip install pandas')
os.system('pip install gluonts')
import pandas as pd
import pathlib
import gluonts
import numpy as np
import argparse
import json
from mxnet import gpu, cpu
from mxnet.context import num_gpus
from gluonts.dataset.util import to_pandas
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer


def train(epochs, prediction_length, num_layers, dropout_rate):
    
    #create train dataset
    df = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TRAIN'] + "/train.csv", header=0, index_col=0)
    
    training_data = ListDataset([{"start": df.index[0], 
                                  "target": df.value[:]}], 
                                  freq="5min")
    
    #define DeepAR estimator
    deepar_estimator = DeepAREstimator(freq="5min",  
                                       prediction_length=prediction_length,
                                       dropout_rate=dropout_rate,
                                       num_layers=num_layers,
                                       trainer=Trainer(epochs=epochs))
    
    #train the model
    deepar_predictor = deepar_estimator.train(training_data=training_data)
      
    #create test dataset
    df = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TEST'] + "/test.csv", header=0, index_col=0)
    
    test_data = ListDataset([{"start": df.index[0], 
                              "target": df.value[:]}], 
                              freq="5min")
    
    #evaluate trained model on test data
    forecast_it, ts_it = make_evaluation_predictions(test_data, deepar_predictor,  num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    
    print("MSE:", agg_metrics["MSE"])
    
    #save the model
    deepar_predictor.serialize(pathlib.Path(os.environ['SM_MODEL_DIR']))
  
    return deepar_predictor


def model_fn(model_dir):
    path = pathlib.Path(model_dir)   
    predictor = Predictor.deserialize(path)
    
    return predictor


def transform_fn(model, data, content_type, output_content_type):
    
    data = json.loads(data)
    df = pd.DataFrame(data)

    test_data = ListDataset([{"start": df.index[0], 
                              "target": df.value[:]}], 
                               freq="5min")
    
    forecast_it, ts_it = make_evaluation_predictions(test_data, model,  num_samples=100)  
    #agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(test_data))
    #response_body = json.dumps(agg_metrics)
    response_body = json.dumps({'predictions':list(forecast_it)[0].samples.tolist()[0]})
    return response_body, output_content_type


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--prediction_length', type=int, default=12)    
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args.epochs, args.prediction_length, args.num_layers, args.dropout_rate)
    
