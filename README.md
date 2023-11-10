# MyoChallenge 2023 - Team Lattice

Team members: Alessandro Marin Vargas, Alberto Chiappa, Alexander Mathis

## Solution ranking 1st in the Relocate task

Here we present the strategy that was employed to achieve the #1 solution to the Relocate task (object manipulation track). We also include the code that was used to train it.

### Structure of the repository

* docker -> Files to create the docker image used to train and test the agents.
* output -> Where the results of the trainings are stored.
* src -> Source code.
  * envs -> Custom environments (also not relevant for this challenge). The important one is the custom version of Relocate.
  * metrics -> Custom callbacks to monitor the trining.
  * models -> Modifications to the RL algorithms, including Lattice exploration.
  * train -> Trainer classes to manage the trainings (not all relevant to this challenge).
  * utils -> Utility functions for the submission.
  * other scripts -> Run the training or test the agent.

### How to run the #1 ranked solution to the object manipulation track

We strongly recommend using docker for maximum reproducibility of the results. We provide the utility scripts `docker/build.sh` and `docker/test.sh` to create a docker image including all the necessary libraries and training/evaluation scripts.

Simply run the script `docker/build.sh` to create a docker image called `myochallengeeval_mani_agent`. The image is fairly large because it was built on top of an image provided by Nvidia to run the library IsaacGym.

Once the image is created, run the script `docker/test.sh` to execute the script `src/test_submission.py` inside a container created from the image `myochallengeeval_mani_agent:latest`. The script `src/test_submission.py` executes 1000 test episode in the environment `myoChallengeRelocateP2-v0` with seed=0. The performance should match exactly the one we obtained, namely, 0.817 (817 episodes solved out of 1000).

By default, the script `src/test_submission.py` tests the last step of the curriculum (from the folder `output/trained_agents/curriculum_step_10`). To test a different pretrained agent, please change the value of the variables `EXPERIMENT_PATH` and `CHECKPOINT_NUM` in the script `src/test_submission.py`. Make sure the checkpoint number corresponds to that of the curriculum step you want to load. Only the curriculum steps 8, 9 and 10 have been trained on the full Relocate task, so we expect the previous checkpoints to perform badly in the full environment.

The script `docker/train.sh` can be used to run a training experiment. We set it so that the training starts from checkpoint 9 of the training curriculum. In the current state, the training will not reproduce the training experiment which lead to checkpoint 10, as the script is not loading the arguments from `output/trained_agents/curriculum_step_10/args.json`. In fact, for the challenge, we used a cluster which accepts parameters in a different format. We did not adapt this part of the code to run in the docker container of the submission. Furthermore, for the challenge trainings we used 250 environments in parallel, requiring substantial RAM resources, unlikely to be available in a standard workstation.

## Summary of our approach

### Key components of the final model

1. On-policy learning with Recurrent PPO
2. Exploration with LATTICE to exploit correlations in the action space
3. Curriculum learning to guide and stabilize training throughout
4. Enlarging the hyperparameter space to achieve a more robust policy

#### 1. Recurrent LSTM layers

The first key component in our model is the recurrent units in both the actor and critic networks of our on-policy algorithm. The first layer in each was an LSTM layer, which was crucial to deal with partial observations not only because the environment had no velocity or acceleration information, but also because the actions have really long-term consequences - effects of actions last dozens of timesteps after it has been originally executed.

#### 2. Exploration with LATTICE

The second key component we used was LATTICE, a new exploration strategy developed by our team. By injecting noise in the latent space, LATTICE can encourage correlations across actuators that are beneficial for task performance and energy efficiency, especially for high-dimensional musculoskeletal models with redundant actuators. Given the complexity of the task, LATTICE allowed us to efficiently explore the state space.

#### 3. Curriculum learning

Third, we used a curriculum of training that gradually increased the difficulty of the task. For both phase 1 and phase 2, we used the same training curriculum steps:

- Reach. Train the agent to minimize the distance between the palm of the hand and the object position. This step could be splitted in substeps such as minimizing first the x-y distance by encouraging the opening of the hand (maximizing the distance between fingertips and palm hand) and then minimizing the z distance as well.
- Grasp & move. Train the agent to minimize the distance between the object position and the target position. In this case, the z-target position was set to be at 40 cm higher than the z-final goal position. Additionally, the x-y position can be the same as the initial object position or can equal the final goal position thereby already minimizing the distance to the target 
- Insert. Train the agent to maximize the solved fraction by inserting the object in the box. While solving the task, we kept (with a lower weight) the part of the reward correlated to the grasp curriculum stage to encourage the policy to continuously try to grasp difficult objects.

Directly transferring the policy of phase 1 to phase 2 was not possible due to the introduction of complex objects and targets. Therefore, we repeated the same curriculum steps with a longer training for phase 2 but we encouraged a more diverse and efficient exploration by using LATTICE. The full list of hyperparameters and links to the models are in the appendix.

#### 4. Enlarging the hyperparameter space

The final insight we tried to incorporate consisted in enlarging the hyperparameter space to obtain a more robust policy. Indeed, we observed that the policy almost reached convergence but it was struggling with objects at the extrame of the range (e.g. small objects). To this end, we made the task harder by increasing the range of shape, friction, mass object hyperparameters. Since part of the reward still consisted to grasp the object and lead it on top of the box, it allowed the policy to continue maximing the task performance while learning to grasp objects at the "extreme" of the hyperparemeter space.
 
For the very final submission that scored 0.343, we used our final robust policy that can be found [here](output/trained_agents/curriculum_step_10/).

Further details about the curriculum steps and architeture can be found in [appendix](docs/appendix.md)

## Further context and literature

If you want to read more about our solution, check out our [NeurIPS work](https://arxiv.org/abs/2305.20065)! 

If you use our code, or ideas please cite:

```
@article{chiappa2023latent,
  title={Latent exploration for reinforcement learning},
  author={Chiappa, Alberto Silvio and Vargas, Alessandro Marin and Huang, Ann Zixiang and Mathis, Alexander},
  journal={arXiv preprint arXiv:2305.20065},
  year={2023}
}
```
