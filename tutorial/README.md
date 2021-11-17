# Parla Tutorial

This tutorial serves as a quick start guide to working with Parla. 
Before starting the tutorial, be sure to install Parla using the instructions in the top-level README. 
Each lesson in this tutorial has its own README which introduces a new feature of Parla. 
Accompanying source code demonstrates that feature.

## Setup

For this tutorial, we provide a Parla container that could be used out of the box. To get a shell inside the provided docker container run

```
docker run --gpus all --rm -it utpecos/parla
```

In this container, a Parla repo with tutorial branch is put at the root of HOME directory.

Depending on your Docker configuration, you may need to run this command as root using sudo or some other method. Since CUDA is required for all the demos, you must provide some GPUs for the docker container to use. For this to work using the command shown, you need to use Docker 19.03 or later.

## Lessons
0. Hello World!  
   - Run your first Parla program.  
1. Intro to Tasks  
   - Create TaskSpaces and Task IDs.  
   - Express dependencies and order between tasks.  
   - Wait on task completion.  
1. More on Tasks  
   - Run tasks in loops.  
   - Call external libraries.  
   - Capture external variables in tasks.  
1. Devices and Architectures  
   - Place tasks on specific devices and architectures.  
   - Create specialized function variants.  
1. Data Movement  
   - Easily move data between devices.  
   - Automatically partition and move data.  
1. Task Constraints  
   - Limit resource usage of tasks.  

## Completion
After completing the tutorial, check out Parla.py/examples to see Parla features used in real applications. 
