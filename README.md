# Experiments

## Environment setup

* Create a virtual environment *myvenv* following the steps in https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/> 
* In `myvenv/bin/activate`, add the following line to reference mujoco: `LD_LIBRARY_PATH="$LD_LIBRARY_PATH/:$HOME/.mujoco/mujoco200/bin"`
* Clone mujoco-py, gym, baselines and experiments from my repo
* Activate *myvenv*: `source myvenv/bin/activate`
* Install:
  * inside experiments: `pip install -r requirements.txt`
  * inside mujoco-py, gym, baselines: `pip install -e .`


## Reproduce results from Plappert et al., 2018 https://arxiv.org/abs/1802.09464

Results in the original paper were produced on a 20-core machine with the following command as stated in this issue: https://github.com/openai/baselines/issues/314

`mpirun -np 19 python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_env=2 --num_timesteps=250000 --save_path="some/path' --log_path="some/path"`

I am currently working on an 8-core machine, which, for some reason, has severe troubles executing `mpirun baselines.run` for `-np > 1` and `--num_env > 1`. Even though, occationally, it works, most of the time, I get errors such as *a system call failed during shared memory initialization that should not have* or *Broken Pipe Error*. When looking at CPU usage, one could see that initiallizing *num_env* environments required a lot of extra computation power, which, together *num_cpu* trainings running in parallel, was to much for *mpirun*.

To understand what is happening, here are some important things to mention:
* `-np <num_cpu>` trainings are run in parallel, the results are combined after each episode
* in each of the trainings, `--num_env=<num_env>` parallel environments are simulated to generate experience (in the HER code, the parameter is used as *rollout_batch_size*, which cannot be changed *config.py*)
* therefore, the overall experience generated is roughly proportional to *<num_cpu>* and *<num_env>*
* `num_epochs = num_timesteps // n_cycles // max_episode_steps // num_env` where num_timesteps is the num_timesteps per cpu!

In order to reproduce the results without using `mpirun`, I kept the product of *<num_cpu>* and *<num_env>* constant and issued the following command on "one core only" with 38 parallel envs. Corresponding number of timesteps: `num_timesteps = num_epochs * n*cycles * max_episode_steps * num_env = 50 * 50 * 50 * 38 = 4750000`.

`python3 -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_env=38 --num_timesteps=4750000 --save_path="some/path' --log_path="some/path"`

This worked out very well, training time was about 2 hours and the results were comparable to the ones in the paper for *FetchPickAndPlace-v1*. Since the parallel envs are basically threads created in the python code, the overall CPU usage was about 75% distributed over all 8 cores on the machine.

## Result figures

The figures that can be found in this directory have all been produced with `num_env=38`. As you can see, *FetchPickAndPlace-v1* results are very similar to the results in the paper. *FetchPush-v1* achieves the same high accuracy, however, it takes slightly longer compared to the paper. Even with doubled number of epochs *FetchSlide-v1* does not achieve the same accuracy as in the paper. Training with even more epochs is underway. In the custom environment *myUR5GripperFall-v1*, the accuracy was not as high as expected, which is due to some hardly predictable bouncing of the stick. A fix of the environment is underway. 


## Environment on Ubuntu16

- had to install some pip packages
- sudo apt-get install libglew-dev
