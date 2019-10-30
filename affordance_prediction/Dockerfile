FROM continuumio/miniconda3

RUN apt-get update --fix-missing && \
    apt-get install -y rclone && \
    apt-get clean

COPY ./environment.yml /
RUN conda env create -f /environment.yml
RUN echo "source activate affordance_training" > ~/.bashrc
ENV PATH /opt/conda/envs/affordance_training/bin:$PATH

COPY ./test.conf /root/.config/rclone/rclone.conf

COPY . /app
WORKDIR /app

ENTRYPOINT ["bash"]
