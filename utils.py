from datetime import datetime

vlevel = None

def vprint(string, required_vlevel, time = True):
	
	if(time):
		string = str(datetime.now().time()) + ' ' + string
	
	if(required_vlevel <= vlevel):
		print(string)
