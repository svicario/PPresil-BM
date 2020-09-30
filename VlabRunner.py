import subprocess
import os
import json

print(os.listdir("./DataInput")")
print(os.listdir("./")")
subprocess.run(["unzip","./DataInput/"]+os.listdir("./DataInput")[0])

try:
    inputfile=[ x for x in os.listdir("./DataInput") if x.split(".")[-1].lower() in ["tif", "tiff", "nc","envi"]][0]
except IndexError:
    print("Input file should have extension name tif, nc or envi")

ARG=json.load(open("vlabparams.json","r"))
arg=""
for k,v in ARG.items():
    if v:
        if v is True:
            v=""
        arg+=" --"+" ".join([k,str(v)])

print(subprocess.run(["python3", "PPresilBayes","--input",inputfile]+arg.split(),stdout=subprocess.PIPE).stdout)


subprocess.run(["zip","Output.zip",ARG.suffix+".nc"])

