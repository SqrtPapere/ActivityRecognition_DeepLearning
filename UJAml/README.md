# Code for UJAml dataset

## To run the code

- Download the dataset at http://ceatic.ujaen.es/ujami/sites/default/files/2018-07/UCAmI%20Cup.zip

- Unzip and copy UCAml Cup&rarr;**Data** in **UJAml** folder of the repo

- There is an error in a header, so: open file at Data&rarr;Training&rarr;2017-11-20&rarr;2017-11-20-C&rarr;**2017-11-20-C-sensors.csv** with any text editor and change the header:

	 *DATE* | OBJECT | STATE | HABITANT          
	--- | --- | --- | ---                                            
	... | ... | ... | ...                         

	to:

	*TIMESTAMP* | OBJECT | STATE | HABITANT          
	--- | --- | --- | ---                                            
	... | ... | ... | ...           


- Download and install all the required modules

- Set parameters:

 	- *window*: is the length in seconds of the desired segmentation (some values won't work because of the architecture of the network!)

	- For the first run we need to create the **labelled version of the data** (*takes a while!*):
		- in multi_input_lstm.py make sure the call to *get_dataset* has **True** arguments:
		
		`data, label, sensors = dataset.get_dataset(directory, window, False, True)`

        - From now on we can set the second Boolean argument to False:
	
		`data, label, sensors = dataset.get_dataset(directory, window, False, False)`
		
		- We can set the first Boolean argument to True, if we already did a segmentation with the same *window* parameter:
	
		`data, label, sensors = dataset.get_dataset(directory, window, True, False)`

