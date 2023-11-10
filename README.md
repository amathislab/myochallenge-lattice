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