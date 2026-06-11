# ==============================================================================
# Custom RabbitMQ image — adds automatic user initialisation on every start.
# ==============================================================================
# The official docker-entrypoint.sh detects whether the CMD starts with
# "rabbitmq*" to decide if it should chown the data dir and drop privileges.
#
# By using a CMD that does NOT start with "rabbitmq", we run as root and
# handle permission fixing + user creation ourselves, then su-exec to the
# rabbitmq user for the actual server process.
# ==============================================================================

FROM rabbitmq:management-alpine

COPY init.sh /workspace/init.sh
COPY startup.sh /usr/local/bin/startup.sh

RUN chmod +x /workspace/init.sh /usr/local/bin/startup.sh

# Keep original ENTRYPOINT (docker-entrypoint.sh).
# Override CMD to a non-"rabbitmq*" name so the entrypoint exec's us as root.
CMD ["/usr/local/bin/startup.sh"]
