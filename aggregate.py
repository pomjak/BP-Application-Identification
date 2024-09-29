import pandas as pd

ds = pd.read_csv('datasets/mobile_desktop_apps_raw.csv', sep=';')

counts = ds['AppName'].value_counts()

print(counts)