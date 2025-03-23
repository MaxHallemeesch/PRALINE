FROM gitlab.ilabt.imec.be:4567/ilabt/gpu-docker-stacks/pytorch-notebook:cuda12-latest

USER root

COPY requirements.txt .

RUN apt-get update \
        && apt-get -y install openslide-tools \
        && apt-get -y install python3-openslide \
        && pip3 install -r requirements_test.txt \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /pre_processing
COPY pre_processing /pre_processing

WORKDIR /main_code
COPY main_code /main_code

WORKDIR /scripts
COPY scripts /scripts

WORKDIR /evaluation
COPY evaluation /evaluation

WORKDIR /spatial_vis
COPY spatial_vis /spatial_vis

WORKDIR /main_code
CMD ["/bin/bash", "/scripts/run_eval_CPTAC_uni.sh"]

#USER jovyan