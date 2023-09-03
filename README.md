# stockpredictor

Stock Predictor

How to run the code:

1- Install the Following Python3 Dependencies:
i- numpy: pip3 install numpy
ii- Pytorch: pip3 install torch
iii- matplotlib: pip3 install matplotlib
iv- alpha_vantage: pip3 install alpha_vantage

2- Setup virtual enviornment and run the following command to install the required dependencies/packages: pip3 install -r requirements.txt

3- Run the getdata.py Script to get the data in the json form, in order to run getdata.py go alpha vantage and generate your api key
and replace it with demo in the url of get_data_for_symbol.

4- After you get the data, run normdata.py to get the normalized values.

5- And then run splitdata.py to split the data in training and validation sets.

6- And in the end run main.py script to use the lstm model on the data to get predictions.
