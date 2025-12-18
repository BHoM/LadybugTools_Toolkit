#The bhom_python_toolkit image must exist in the docker image container, see https://github.com/BHoM/Python_Toolkit.
FROM bhom_python_toolkit:latest

#Install ladybugtools_toolkit packages.
COPY . /LadybugTools_Toolkit

RUN pip3 install --no-cache-dir ./LadybugTools_Toolkit
RUN rm ./LadybugTools_Toolkit -rf

RUN apt-get update \
    && apt-get install wget -y

#Install openstudio and dependencies.
RUN wget https://github.com/NREL/OpenStudio/releases/download/v3.10.0/OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb \
    && dpkg-deb -R OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb . \
    && rm OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb \
    && apt install libgomp1 libx11-6 -y

#install radiance
RUN wget https://github.com/LBNL-ETA/Radiance/releases/download/rad5R4/Radiance_4ee32974_Linux.zip \
    && apt-get install unzip -y \
    && unzip Radiance_4ee32974_Linux.zip \
    && tar --strip-components=1 -zxf radiance-5.4.4ee32974b1-Linux.tar.gz \
    && rm radiance-5.4.4ee32974b1-Linux.tar.gz \
    && rm Radiance_4ee32974_Linux.zip

#remove install packages and clean up
RUN apt-get remove wget -y \
    && apt-get remove unzip -y \
    && apt autoremove -y

#add radiance directory to PATH
ENV RAYPATH="/usr/local/radiance/lib"