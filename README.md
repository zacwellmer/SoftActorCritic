# Soft Actor Critic
This is an implementation of [Soft Actor Crtitic](https://arxiv.org/pdf/1801.01290.pdf) (The official implementation can be found [here](https://github.com/rail-berkeley/softlearning)) with Tensorflow 2.1. 
It is partially based off [OpenAI's spinningup Soft Actor Critic](https://github.com/openai/spinningup). 

# Setup
The easiest setup is with Docker.
```
docker image build -t sac:1.0 -f docker/Dockerfile.sac .
docker container run --detach -it --name sac sac:1.0
docker attach sac
```
