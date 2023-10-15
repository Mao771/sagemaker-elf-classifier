import glob
import os.path
import pandas as pd
import argparse
import joblib
from io import StringIO

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

os.environ.setdefault('OPCODES_COLUMN_NAME', 'processed_opcodes')
os.environ.setdefault('TARGET_COLUMN_NAME', 'virus')
os.environ.setdefault('SM_MODEL_DIR', 'model')
os.environ.setdefault('SM_CHANNEL_TRAIN', 'train')

MODEL_NAME = "sgd_elf_classifier.joblib"


def model_fn(model_dir):
    print("Model function")
    try:
        clf = joblib.load(os.path.join(model_dir, MODEL_NAME))
    except Exception:
        print("Model not found")
        clf = None
    return clf


def process_data(data):
    return data


def transform_data(data):
    return data


def input_fn(request_body, request_content_type):
    print("Input function")
    if request_content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body))
        return process_data(data)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise ValueError("{} not supported by script!".format(request_content_type))


def predict_fn(input_data, model):
    print("Predict function")
    X = transform_data(input_data)
    print(X)
    prediction = model.predict(X)
    return {'predicted-value': prediction}


def prepare_training_data(training_files):
    data = pd.DataFrame()
    for file in training_files:
        data.append(pd.read_csv(file))


def output_fn(prediction, accept):
    print("Output function")
    return prediction


def run_training(args):
    print("Training SGD classifier")
    training_files = glob.glob(os.path.join(args.train, "*.csv"))
    sgd = SGDClassifier(loss="log_loss", verbose=1, average=True)

    for training_file in training_files:
        print(training_file)
        train_df = pd.read_csv(training_file)

        train_x = train_df[args.opcodes_column]
        train_y: pd.Series = train_df[args.target_column]

        count_vect = CountVectorizer(ngram_range=(4, 4))
        X_train_counts = count_vect.fit_transform(train_x)
        tf_idf_vectorizer = TfidfTransformer()
        X_train_tfidf = tf_idf_vectorizer.fit_transform(X_train_counts)

        print(train_y.value_counts())

        sgd.fit(X_train_tfidf, train_y)

    joblib.dump(sgd, os.path.join(args.model_dir, MODEL_NAME))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--opcodes-column', type=str, default=os.environ.get('OPCODES_COLUMN_NAME'))
    parser.add_argument('--target-column', type=str, default=os.environ.get('TARGET_COLUMN_NAME'))
    args, _ = parser.parse_known_args()
    # another function that does the real work
    # (and make the code cleaner)
    run_training(args)
