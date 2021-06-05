FROM ubuntu:latest

###### install SEAL dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update; \ 
    apt-get -y install git wget awscli build-essential clang-tools-9 libmsgsl-dev zlib1g-dev cmake

###### build and install SEAL
RUN git clone --branch 3.6.5 https://github.com/microsoft/SEAL.git && \
    cd SEAL; \
    cmake -S . -B build; \
    cmake --build build; \
    cmake --install build

###### install other ABC dependencies
RUN apt-get update; \
    apt-get -y install nlohmann-json3-dev

###### run the passed benchmark
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
