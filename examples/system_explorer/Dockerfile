
FROM python:3.12-bookworm

RUN wget -qO- https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

CMD ["/bin/bash"]
