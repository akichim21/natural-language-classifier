FROM yamitzky/miniconda-neologd

COPY env.yaml /tmp
COPY app /tmp

# mecabがgcc必用
RUN apt-get install -y build-essential

RUN conda env create --file /tmp/env.yaml
WORKDIR /tmp
