from shutil import SpecialFileError
from utils.all_utils import *
from utils.plot import *
from utils.ann_model import *
from utils.cnn_model import *
import argparse

def training(config_path):
    config = read_config(config_path)

    file_path = config['params']['file_path']
    df = get_data(file_path)

    unnecessary_columns = config['params']['unnecessary_columns']
    df_new = remove_unnecessary_columns(df, unnecessary_columns)
    print(config)


    df_new = drop_missing_columns(df_new)

    df_new = drop_zero_columns(df_new)

    x, y = split_data(df_new)

    x = remove_correlated_columns(x, 0.90)

    print(f"x data shape - {x.shape}")

    print(f"y data shape - {y.shape}")

    x = minmaxscaler_scaling(x)


    x = x.reshape(-1, x.shape[1], 1)

    y = y.to_numpy()

    

    test_size = config['params']['test_size']
    x_train, x_test, y_train, y_test = split_train_test_data(x,y, test_size)

    #loss_function = config['params']['loss_function']

    loss = config["params"]["loss_function"]
    optimizer = config["params"]["optimizer"]
    num_classes = config["params"]["num_classes"]
    kernel_size = config["params"]["kernel_size"]

    model = cnn_model(x_train, loss, optimizer, num_classes, kernel_size )

    EPOCHS = config["params"]["epochs"]
    BATCH_SIZE = config["params"]["batch_size"]
    validation_data = config["params"]["validation_datasize"]
    print(validation_data)
    verbose = config["params"]["verbose"]
    
    model.fit(x_train, y_train, epochs = EPOCHS, validation_split= validation_data, batch_size=BATCH_SIZE, verbose=verbose)

  
  #  save_plot(y_test, pred)
    



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", "-c", help="Path to config file", default="config.yaml")
 
    parsed_args = args.parse_args()
  
    training(config_path=parsed_args.config)