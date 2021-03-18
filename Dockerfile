FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip 
RUN python3 -m pip install --upgrade pip 
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntugis/ppa
RUN apt-get update
    #apt-get install -y gdal-bin=2.2.2+dfsg-1~xenial1 python3-gdal=2.2.2+dfsg-1~xenial1 &&\
RUN pip3 install pandas && \
    pip3 install scipy && \
    pip3 install numpy && \
    pip3 install sklearn && \
    pip3 install patsy && \
    pip3 install rasterio && \
    pip3 install xarray && \
    pip3 install pyproj && \
    pip3 install netcdf4 && \
    pip3 install geopandas && \
    pip3 install salem

RUN apt-get install -y zip && \
    apt-get install -y unzip && \
    apt-get install -y gdal-bin && \
    apt-get install -y awscli
