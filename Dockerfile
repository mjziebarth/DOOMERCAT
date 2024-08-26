FROM quay.io/pypa/manylinux_2_28_x86_64
WORKDIR /doomercat
COPY LICENSE meson.build pyproject.toml README.md setup.py \
    ./
COPY subprojects/boost-subset.wrap ./subprojects/
COPY subprojects/packagefiles/* ./subprojects/packagefiles/
COPY doomercat/ ./doomercat/
COPY cpp/ ./cpp/

RUN \
    export MANYLINUX_PLATFORM_TAG="manylinux_2_28_x86_64"; \
    ls /opt/python/cp313-cp313/bin/; \
    #/opt/python/cp313-cp313/bin/pip install wheel; \
    /opt/python/cp313-cp313/bin/pip wheel .

RUN \
    ls;