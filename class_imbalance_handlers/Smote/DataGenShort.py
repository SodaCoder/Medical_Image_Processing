import pandas as pd
import os
import shutil

def CreateCSVFile(srcPath, dstPath):
    covidFiles = os.listdir(srcPath + "/COVID")
    normalFiles = os.listdir(srcPath + "/Normal")
    pneumoniaFiles = os.listdir(srcPath + "/Pneumonia")
    lungOpacityFiles = os.listdir(srcPath + "/Lung_Opacity")
    tbFiles = os.listdir(srcPath + "/TB")

    covidLabels = [0] * len(covidFiles)
    normalLabels = [1] * len(normalFiles)
    pneumoniaLabels = [2] * len(pneumoniaFiles)
    lungOpacityLabels = [3] * len(lungOpacityFiles)
    tbLabels = [4] * len(tbFiles)

    diseaseFiles = [*covidFiles, *normalFiles, *pneumoniaFiles, *lungOpacityFiles, *tbFiles]
    diseaseLabels = [*covidLabels, *normalLabels, *pneumoniaLabels, *lungOpacityLabels, *tbLabels]

    diseaseDict = {"images" : diseaseFiles, "labels" : diseaseLabels}
    df = pd.DataFrame(diseaseDict)
    df.to_csv(dstPath, columns = ["images", "labels"], index = False)

    print("Writing to CSV file done...")

def CopyDir(srcPath, dstPath):
    covidDir = srcPath + "/COVID"
    normalDir = srcPath + "/Normal"
    pneumoniaDir = srcPath + "/Pneumonia"
    lungOpacityDir = srcPath + "/Lung_Opacity"
    tbDir = srcPath + "/TB"
    
    shutil.copytree(covidDir, dstPath, dirs_exist_ok=True)
    shutil.copytree(normalDir, dstPath, dirs_exist_ok=True)
    shutil.copytree(pneumoniaDir, dstPath, dirs_exist_ok=True)
    shutil.copytree(lungOpacityDir, dstPath, dirs_exist_ok=True)
    shutil.copytree(tbDir, dstPath, dirs_exist_ok=True)
    print("Copying to all_images done...")

#CreateCSVFile("../5classPreprocessed/train", "../5classPreprocessed/train.csv")
#CreateCSVFile("../5classPreprocessed/val", "../5classPreprocessed/val.csv")
#CreateCSVFile("../5classPreprocessed/test", "../5classPreprocessed/test.csv")
dstPath = "../Output_5cls/all_images"
CopyDir("../Output_5cls/train", dstPath)
CopyDir("../Output_5cls/val", dstPath)
CopyDir("../Output_5cls/test", dstPath)
