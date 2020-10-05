FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    python3 -m pip install --upgrade pip && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntugis/ppa && \
    apt-get update && \
    #apt-get install -y gdal-bin=2.2.2+dfsg-1~xenial1 python3-gdal=2.2.2+dfsg-1~xenial1 &&\
    pip3 install pandas && \
    pip3 install scipy && \
    pip3 install numpy && \
    pip3 install sklearn && \
    pip3 install patsy && \
    pip3 install rasterio && \
    pip3 install xarray && \
    pip3 install pyproj && \
    pip3 install netcdf4 && \
    pip3 install pyyaml && \
    apt-get install -y zip && \
    apt-get install -y unzip
