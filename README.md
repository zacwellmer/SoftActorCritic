# Soft Actor Critic
This is an implementation of [Soft Actor Crtitic](https://arxiv.org/pdf/1801.01290.pdf) (The official implementation can be found [here](https://github.com/rail-berkeley/softlearning)) with Tensorflow 2.1. 
It is partially based off [OpenAI's spinningup Soft Actor Critic](https://github.com/openai/spinningup). 

# Setup
The easiest way to setup is with Docker after cloning the repo.
```
docker image build -t sac:1.0 -f docker/Dockerfile.sac .
docker container run --detach -it --name sac sac:1.0
docker attach sac
```

# Run
```
python sac.py --env Pendulum-v0 --hid 256 --l 2 --gamma 0.99 --epochs 50 --exp_name magic_sac
```
Will run Soft Actor Critic on Pendulum-v0 with 2 hidden layers each with 256 units for 50 epochs. Logging will be stored under name "data/magic_sac"
