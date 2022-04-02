import shutil
import os

def moveFile(srcDir, dstDir):
    file_names = os.listdir(srcDir)

    for file_name in file_names:
        shutil.move(os.path.join(srcDir, file_name), dstDir)

    print("Moving from ", srcDir, " to ", dstDir, " is done...")

dstDir = "./all_images"
srcDir = ["./Preprocessed", "./Test", "./Val"]
srcSubDir = ["COVID", "Pneumonia", "Normal", "Lung_Opacity"]

fullSrcDir = []
for outerDir in srcDir:
    for innerDir in srcSubDir:
        fullSrcDir = fullSrcDir + [os.path.join(outerDir, innerDir)]
print(fullSrcDir)

for item in fullSrcDir:
    moveFile(item, dstDir)
