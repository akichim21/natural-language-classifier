FROM yamitzky/miniconda-neologd

WORKDIR /tmp

# mecabがgcc必用
RUN apt-get install -y build-essential

COPY env.yaml /tmp
RUN conda env create --file /tmp/env.yaml

ENTRYPOINT /bin/bash