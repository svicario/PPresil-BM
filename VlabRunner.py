import subprocess
import os
import json
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd

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
if "mask.zip" in os.listdir("./"):
    subprocess.run(["unzip","mask.zip","-d","./MASK"])
    try:
        L=os.listdir("./MASK")
    except FileNotFoundError:
        ARG["mask"]="false"
    else:
        if len(L)==1:
            path="./MASK/"+L[0]
        else:
            path="./MASK/"+[ x for x in L if x.split(".")[-1].lower() in ["envi","shp"]][0]
    try:
        V=gpd.read_file(path)
    except Exception as inst:
        if str(type(inst))!="<class 'fiona.errors.DriverError'>":
            raise(KeyError)
        ARG["mask"]="false"
    else:
        ARG["mask"]=path
print("ARG")
print(ARG)
arg=""
for k,v in ARG.items():
    if (v is False)|(v in ["False","false","F"]):
        continue
    else:
        if (v is True)|(v=="true"):
            v=""
        arg+=" --"+" ".join([k,str(v)])

print("arguments")
print(arg)
arg=""
for k,v in ARG.items():
    if (v is False)|(v in ["False","false","F"]):
        continue
    else:
        if (v is True)|(v=="true"):
            v=""
        arg+=" --"+" ".join([k,str(v)])

print("arguments")
print(arg)


subprocess.run(["python3", "PPresilBayes.py","--input","./DataInput/"+inputfile]+arg.split(),stdout=subprocess.PIPE).stdout


#WMS
import xarray as xr
A=xr.open_dataset(ARG["suffix"]+".nc")
A["sdinter_mean"]*=10000
A["sdinter_mean"]=A["sdinter_mean"].astype("int16")
A["sdinter_mean"].to_netcdf(ARG["suffix"]+"sdinter_mean.nc")
Min,Max=A.sdinter_mean.quantile(q=[0.05,0.975]).values
bins=10
Z=pd.read_csv("paletteViridis.csv", header=None)
Z[0]=np.linspace(Min,Max,bins).astype("int")
Z.to_csv("paletteViridisMod.csv", index=False, header=False)


CMD="gdaldem color-relief -alpha -nearest_color_entry NETCDF:{suffix}sdinter_mean.nc:sdinter_mean paletteViridisMod.csv {suffix}sdinter_mean.png".format(suffix=ARG["suffix"])
subprocess.run(CMD.split())

#CMD="rgb2pct.py {suffix}sdinter_mean.png {suffix}sdinter_mean_pct.png".format(suffix=ARG["suffix"])
#subprocess.run(CMD.split())
#gdalwarp -s_srs_
#subprocess.run(["gdal_translate", 'NETCDF:'+ARG["suffix"]+".nc:sdinter_mean", ARG["suffix"]+"sdinter_mean.tif"])




subprocess.run(["gdal2tiles.py", "-z","2-14", "-c", "CNR-IIA/vicario 2020", "-s",  A.pyproj_srs, ARG["suffix"]+"sdinter_mean.png"])


ID=os.environ['BENGINERUNID']
print(ID)
CMD='/root/.local/bin/aws s3 sync {suffix}sdinter_mean s3://testtilebucket/{ID}{V1}/3857 --acl public-read'.format(ID=ID, V1="V1",suffix=ARG["suffix"])
subprocess.run(CMD.split())

D={"url":"https://eddipf6en8.execute-api.us-east-1.amazonaws.com/alpha/wms?", "protocol":"urn:ogc:serviceType:WebMapService:1.1.1:HTTP", "name":ID+"V1"} 
with open('wms.json', 'w') as outfile:
    json.dump(D, outfile)

print(D)

subprocess.run(["zip","Output.zip",ARG["suffix"]+".nc"])
