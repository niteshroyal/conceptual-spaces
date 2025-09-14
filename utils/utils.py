import os
import logging
import matplotlib.pyplot as plt


def read_txt_files_from_folder(folder_path):
    all_files = os.listdir(folder_path)
    txt_files = [file for file in all_files if file.endswith('.txt')]

    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        read_data(file_path)

    pass


def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    headers = lines[0].strip().split('\t')

    data = []
    for line in lines[1:]:
        values = line.strip().split('\t')
        if values == ['']:
            pass
        else:
            data.append(dict(zip(headers, values)))

    return headers, data


def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    logging.info(f'Number of training + validation datapoints = {len(lengths)}')

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    plt.title('Distribution of number of tokens')
    plt.show()
