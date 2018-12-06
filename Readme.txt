------------Download data from GDC and data preparation------

1. The steps of downloading GDC dataset can be found in GDC lab.pdf

2. install all the packages needed. (sklearn, pandas)

pip install sklearn 
pip install pandas
Pip install matplotlib

3. Source codes:
check.py:
		 This is to check the integrity for the downloaded RNA files
		 python check.py. You need to change directory in code. 

parse_file_case_id.py:  
		This is to get the unique file id and the corresponding case ids.You need to change directory in code. 	
		python parse_file_case_id.py

Transpose.py:
		The GTex data are different from GDC data, to make sure they are accordingly related, use the .py to transpose the GTex data.You need to change directory in code. 
		python Transpose.py

request_meta.py: This is to request the meta data for the files and cases.You need to change directory in code. 
		python request_meta.py

gen_miRNA_matrix.py: This is to generate the miRNA matrix and labels for all the files.You need to change directory in code. 
		python gen_miRNA_matrix.py

GDC&GTex, BreastReadCount-Separate and Save.ipynb: This will run on Amazon SageMaker, use this file to seprate GDC data and GTex data and use that for K means clustering later.

KMeans On Each dataset.ipynb: This will run on Amazon SageMaker, use this file to run KMeans clustering and see the result of clustering.

MixedData-PCA-TSNE.ipynb: This will run on Amazon SageMaker, use this file to perform PCA and t-SNE and see the result

ApplyMachineLearning.ipynb: This will run on Amazon SageMkaer, use this file to perfrom Machine Learning and see result from mixed data

predict.py : This is for applying models to the mixed cancer types matrix for tumor sample detection.You need to change directory in code. 
		python predict.py

