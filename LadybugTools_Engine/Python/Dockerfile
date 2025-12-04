#The bhom_python_toolkit image must exist in the docker image container, see https://github.com/BHoM/Python_Toolkit.
FROM bhom_python_toolkit:latest

#Install ladybugtools_toolkit packages.
COPY . /LadybugTools_Toolkit

RUN pip3 install --no-cache-dir ./LadybugTools_Toolkit
RUN rm ./LadybugTools_Toolkit -rf

#Install openstudio and dependencies.
RUN apt-get update \
    && apt-get install -y wget \
    && wget https://github.com/NREL/OpenStudio/releases/download/v3.10.0/OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb \
    && dpkg-deb -R OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb . \
    && rm OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb -y \
    && apt install libgomp1 libx11-6 -y \
    && apt-get remove wget -y \
    && apt autoremove -y
