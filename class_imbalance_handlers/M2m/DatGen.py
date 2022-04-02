import pandas as pd
import os

def CreateCSVFile(srcPath, dstPath):
    covidFiles = os.listdir(srcPath + "/COVID")
    lungOpFiles = os.listdir(srcPath + "/Lung_Opacity")
    normalFiles = os.listdir(srcPath + "/Normal")
    pneumoniaFiles = os.listdir(srcPath + "/Pneumonia")

    covidLabels = [0] * len(covidFiles)
    lungOpLabels = [1] * len(lungOpFiles)
    normalLabels = [2] * len(normalFiles)
    pneumoniaLabels = [3] * len(pneumoniaFiles)

    diseaseFiles = [*covidFiles, *lungOpFiles, *normalFiles, *pneumoniaFiles]
    diseaseLabels = [*covidLabels, *lungOpLabels, *normalLabels, *pneumoniaLabels]

    diseaseDict = {"images" : diseaseFiles, "labels" : diseaseLabels}
    df = pd.DataFrame(diseaseDict)
    df.to_csv(dstPath, columns = ["images", "labels"], index = False)

    print("Writing to CSV file done...")

CreateCSVFile("./Preprocessed", "./train.csv")
CreateCSVFile("./Val", "./val.csv")
CreateCSVFile("./Test", "./test.csv")
