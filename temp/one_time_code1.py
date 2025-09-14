import pandas as pd

file_name = '/home/nitesh/research/conceptual-spaces/data/merged_MusicDataset.csv'

output_file_name = '/home/nitesh/research/conceptual-spaces/data/Ranking_DataSet/Music/music_perception.txt'

df = pd.read_csv(file_name, sep=',', encoding='utf-8')

df['artist - title'] = df['artist'] + ' - ' + df['title']

df[['artist - title', 'wond', 'tran', 'tend', 'nost', 'peac', 'joya', 'ener', 'sadn', 'tens']].to_csv(output_file_name, sep='\t', index=False, header=True)



# file_name = '/home/nitesh/research/conceptual-spaces/data/odour_dataset_4features.xlsx'

# output_file_name = '/home/nitesh/research/conceptual-spaces/data/Ranking_DataSet/Odour/odour_dataset.txt'

# df = pd.read_excel(file_name, sheet_name='Sheet1')

# df['items'] = df['Odor']
# df[['items', 'familiarity', 'intensity', 'pleasantness', 'irritability']].to_csv(output_file_name, sep='\t', index=False, header=True)

