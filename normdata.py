import json
import numpy as np
import matplotlib.pyplot as plt



def normalize_data(input_file, output_file):
    #Reads file, Normalizes the data and writes the adjusted close values in the new norm file.
    
    with open(f"./temp/{input_file}.json", 'r') as f:
        data = json.load(f)
    f.close()

    #Takes the adjusted close values from the data and finds its mean and standard deviation
    adjusted_close = [float(values["5. adjusted close"]) for values in data.values()]
    mu = np.mean(adjusted_close)
    std = np.std(adjusted_close)

    #Normalizes adjusted close values
    norm_data = {}

    for date, values in data.items():
        adjusted_close_values = float(values.get("5. adjusted close"))
        if adjusted_close_values:
            normalized_adjusted_close = normalize_value(float(adjusted_close_values), mu, std)
            norm_data[date] = {"5. adjusted close": normalized_adjusted_close}

    with open(f"./temp/{output_file}.json", 'w') as f:
        json.dump(norm_data, f, indent=4, separators=(',', ": "))
    f.close()


def plot_graph(file_name):
    #Reads normalized file and Plots dates vs values
    
    dates = []
    normalized_values = []

    with open(f"./temp/{file_name}.json", "r") as f:
        data = json.load(f)
        for date, values in data.items():
            dates.append(date)
            normalized_values.append(values["5. adjusted close"])
    f.close()

    plt.plot(dates, normalized_values, color = 'red', linestyle='--')
    plt.title("Adjusted close value vs Year")
    plt.xlabel('year', fontsize=14)
    plt.ylabel('adjusted close value', fontsize=14)
    plt.xticks(np.arange(0,len(dates)), rotation='vertical')
    plt.grid(True)
    plt.show()



def normalize_value(value , mu, std):
    if std == 0:
        return 0
    else:
        return (value - mu) / std


def main():
    input_file = "IBM"
    output_file = (f"{input_file}norm")
    normalize_data(input_file, output_file)
    plot_graph(output_file)


if __name__ == "__main__":
    main()
