# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
import csv 
from utils import logger
def file_as_bytes(file):
    with file:
        return file.read()


def get_cancer_dic(file):
	# 读取csv至字典
	csvFile = open(file, "r")
	reader = csv.reader(csvFile)
	# 建立空字典
	myset = set()
	res = 0 
	for item in reader:
	    # 忽略第一行
	    if reader.line_num == 1:
	        continue
	    myset.add(item[-1])

	# example: result["lung"] = num
	csvFile.close()

	cancer_type = list(myset)
	cancer_type_dic = {}

	for i, val in enumerate(cancer_type, start = 1):
		cancer_type_dic[val] = i

	# if no tumor, value is 0
	cancer_type_dic["no"] = 0
	return cancer_type_dic

def extractMatrix(dirname):
	'''
	return a dataframe of the miRNA matrix, each row is the miRNA counts for a file_id

	'''
	count = 0

	miRNA_data = []
	for idname in os.listdir(dirname):
		# list all the ids 
		if idname.find("-") != -1:
			idpath = dirname +"/" + idname

			# all the files in each id directory
			for filename in os.listdir(idpath):
				# check the miRNA file
				if filename.find("-") != -1:

					filepath = idpath + "/" + filename
					df = pd.read_csv(filepath,sep="\t")
					# columns = ["miRNA_ID", "read_count"]
					if count ==0:
						# get the miRNA_IDs 
						miRNA_IDs = df.miRNA_ID.values.tolist()

					id_miRNA_read_counts = [idname] + df.read_count.values.tolist()
					miRNA_data.append(id_miRNA_read_counts)

					count +=1
					# print (df)
	columns = ["file_id"] + miRNA_IDs
	df = pd.DataFrame(miRNA_data, columns=columns)
	return df

def extractLabel(inputfile):
	df = pd.read_csv(inputfile, sep="\t")
	mydic = get_cancer_dic("miRNA_matrix.csv")
	# print (df[columns])
	print(mydic)
	def number_to_flag(primary_site):
		if primary_site == "Normal":
			return mydic["no"]
		else:
			return mydic[primary_site]

	df['label'] = df['cases.0.samples.0.sample_type']
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'label'] = 0
	for col in df[df['cases.0.samples.0.sample_type'].str.contains("Normal") == True].index.tolist():
		print(col)
		df.loc[col, "label"] = 2
	# df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal") == False, 'label'] = mydic[df["file_id"].values[0]]
	# df["label"] = df['cases.0.samples.0.sample_type'].map(number_to_flag)
	tumor_count = df.loc[df.label == 1].shape[0]
	normal_count = df.loc[df.label == 0].shape[0]
	logger.info("{} Normal samples, {} Tumor samples ".format(normal_count,tumor_count))
	columns = ['file_id','label']
	return df[columns]

if __name__ == '__main__':


	data_dir ="/Users/Lxc/Desktop/Cloud_Computing/lab10/"
	# Input directory and label file. The directory that holds the data. Modify this when use.
	dirname = data_dir + "live_miRNA"
	label_file = data_dir + "files_meta.tsv"
	
	#output file
	outputfile = data_dir + "miRNA_matrix1.csv"

	# extract data
	matrix_df = extractMatrix(dirname)
	label_df = extractLabel(label_file)

	#merge the two based on the file_id
	result = pd.merge(matrix_df, label_df, on='file_id', how="left")
	#print(result)

	#save data
	result.to_csv(outputfile, index=False)
	#print (labeldf)

 




