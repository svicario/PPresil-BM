import subprocess
import os
import json


#print(os.listdir("./DataInput"))
print(os.listdir("./"))
#subprocess.run(["unzip","./DataInput/"]+os.listdir("./DataInput")[0])
subprocess.run(["unzip","DataInput.zip", "-d","./DataInput"])
print(os.listdir("./"))
try:
    inputfile=[ x for x in os.listdir("./DataInput") if x.split(".")[-1].lower() in ["tif", "tiff", "nc","envi"]][0]
except IndexError:
    print("Input file should have extension name tif, nc or envi")

ARG=json.load(open("vlabparams.json","r"))
print("ARG")
print(ARG)
arg=""
for k,v in ARG.items():
    if v:
        if v is True:
            v=""
        arg+=" --"+" ".join([k,str(v)])

print("arguments")
print(arg)
print(subprocess.run(["python3", "PPresilBayes.py","--input",inputfile]+arg.split(),stdout=subprocess.PIPE).stdout)


subprocess.run(["zip","Output.zip",ARG.suffix+".nc"])

