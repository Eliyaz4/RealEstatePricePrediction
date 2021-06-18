import pickle
import json
import numpy as np

locations=None
data_columns=None
model=None
def get_location_names():
	return locations
def get_estimated_price(location,sqft,bhk,bath):
	try:
		loc_index=data_columns.index(location.lower())
	except:
		loc_index=-1
	x=np.zeros(len(data_columns))
	x[0]=sqft
	x[1]=bath
	x[2]=bhk
	if(loc_index>=0):
		x[loc_index]=1
	return round(model.predict([x])[0],2)
def load():
	global locations
	global data_columns
	global model
	with open('./columns.json','r') as f:
		data_columns=json.load(f)['data_columns']
		locations=data_columns[3:]
	with open('./bangalore_land.pickle','rb') as f:
		model=pickle.load(f)
load()
