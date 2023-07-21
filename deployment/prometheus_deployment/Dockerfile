FROM prom/prometheus:v2.21.0

COPY ./prometheus /etc/prometheus/
WORKDIR /prometheus

EXPOSE 9090

ENTRYPOINT [ "/bin/prometheus" ]
CMD [ "--config.file=/etc/prometheus/prometheus.yml", \
    "--storage.tsdb.path=/prometheus", \
    "--web.enable-lifecycle"]