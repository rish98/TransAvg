import torch
import matplotlib.pyplot as plt

import numpy as np
import scipy
# import matplotlib.pyplot as plt
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Generate two random tensor matrices
# tensor1 = torch.load("./clean_params/params_40.pt").cpu()
# # tensor2 = torch.load("./mal_neg_params/params_10.pt")

# # Flatten the tensor matrices and sort the values
# flattened1 = torch.flatten(tensor1.weight).detach().numpy()
# # flattened2 = torch.flatten(tensor2)

# norm_cdf = scipy.stats.norm.cdf(flattened1) # calculate the cdf - also discrete

# # plot the cdf
# sns.lineplot(x=flattened1, y=norm_cdf)
# plt.show()

# Initialize an empty dictionary to store the data
data_dict = {}

# Open the file for reading
with open('../graphs.txt', 'r') as f:
    # Iterate over each line in the file
    for line in f:
        # Split the line by space
        parts = line.split()
        
        # Extract the key (first characters before the space)
        key = parts[0]
        
        # Extract the values (remaining parts after the space)
        values = []
        for value_str in parts[1:]:
            # Split each value by commas and convert to float
            value_list = [float(val) for val in value_str.split(',') if val]  # Remove empty strings before conversion
            values.extend(value_list)  # Extend the values list with the converted floats
        
        # Store the key-value pair in the dictionary
        data_dict[key] = values

# Print the dictionary
print((data_dict.keys()))

import matplotlib.pyplot as plt

# Three lists of accuracy values (replace these with your actual data)
accuracy_list1 = data_dict["FedAvg-baseline"]
accuracy_list2 = data_dict["FedAvg-inverse(5)"]
accuracy_list3 = data_dict["FedAvg-noise_0.1(5)"]
accuracy_list4 = data_dict["FedAvg-flip_1-9(5)"]

# Generate x-axis values (assuming each list represents accuracy at different steps)
x_values = list(range(0, 101))  # Assuming 100 steps

# Plot the three lists on the same graph
plt.plot(x_values, accuracy_list1, label='FedAvg-baseline')
plt.plot(x_values, accuracy_list2, label='FedAvg-inverse(5)')
plt.plot(x_values, accuracy_list3, label='FedAvg-noise_0.1(5)')
plt.plot(x_values, accuracy_list4, label='FedAvg-flip_1-9(5)')

# Add labels and legend
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()



