import requests
import json
import datetime


def get_data_for_symbol(symbol):
    #Creates URL for the symbol and fetches data for it. Returns JSON formatted data.
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey=L7QNGXR1XCX8OCA8'
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.HTTPError as errh:
        print("HTTP ERROR!")
        print(errh.args[0])

    return data


def filter_data(data, cutoff_date):
    #Given the cutoff date, this filters data. Return a dict.
    cutoff_date = datetime.datetime(2019, 11, 1)
    data_dict = {}
    for d in data['Time Series (Daily)']:
        date_split = d.split('-')
        current_date = datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2]))
        if current_date <= cutoff_date:
            data_dict.update({d: data['Time Series (Daily)'][d]})
    
    return data_dict


def write_json(data, symbol):
    f= open(f"./temp/{symbol}.json", "w")
    f.seek(0)
    json.dump(data, f, indent=4, separators=(',', ': '))
    f.close()
    print(f"Successfully Wrote {symbol}.json!!!")

def main():
    symbol = "IBM"
    cutoff_date = datetime.datetime(2019, 11, 1)
    data = get_data_for_symbol(symbol)
    filtered_data = filter_data(data, cutoff_date)
    write_json(filtered_data, symbol)


if __name__ == "__main__":
   main()