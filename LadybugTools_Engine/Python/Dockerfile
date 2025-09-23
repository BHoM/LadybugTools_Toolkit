FROM bhom_python_toolkit:latest

COPY . /LadybugTools_Toolkit

RUN pip3 install ./LadybugTools_Toolkit

RUN apt-get update && apt-get install -y wget

RUN wget https://github.com/NREL/OpenStudio/releases/download/v3.10.0/OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb

RUN dpkg-deb -R OpenStudio-3.10.0+86d7e215a1-Ubuntu-22.04-x86_64.deb .

run apt-get remove wget -y

run apt autoremove -y

RUN rm ./LadybugTools_Toolkit -rf