# Arnold - The Muscle Transformer

**This repo uses a modified version of stable_baselines3**

## Run cpu-based tasks

## Run gpu-based tasks

Please include `--using_isaac` when running on tasks from isaacgym. One example may look like:

```bash
python main_arnold.py --env_names IsaacgymHandReach --num_envs=64 --using_isaac --using_tensor_buffer --headless
```

If you have a limited gpu-ram, then you may want to store rollouts on cpu-ram. Removing `--using_tensor_buffer` will do the job.

```bash
python main_arnold.py --env_names IsaacgymHandReach --num_envs=64 --using_isaac --headless
```

Please note, without `--using_tensor_buffer` command, there will be frequent copy between gpu-ram and cpu-ram, thus reducing the performance.