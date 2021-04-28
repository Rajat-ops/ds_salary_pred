import scrap as sc
import pandas as pd

path = "F:\\chromedriver.exe"

df = sc.get_jobs('data_scientist', 15, False, path, 15)

df.to_csv('glassdoor_jobs.csv', index = False)