## Three cognitive primitive evaluations

### Before or behind
Use a transparent glass and a goal. Place the goal behind the glass or before the glass. Classify.
Let agent take five steps for the message to propagate. Freeze and do DIR.

### Occlusion
Move object behind the occluding wall.
Classify at different frames whether the object is occluded.
Strapped observation.

### Rotation permanence
Find object.

Rotate 180.

Take four more actions: set the last few features as positive example.

For negative example, set the object to be invisible the entire time. Also take the last few features.

## General methodology
To create the evaluations to train MLP from, we
recognize a few constraints.

1) The environment yml needs to be procedurally generated to ensure diversity.

2) Data points are collected from running of a fixed agent.
   The agent's internal representations need to be retrieved to generate
   the dataset.
   
3) Visibility is achieved as the environment is defined. It has to be.

## todo 3/14
Okay we are not using ray. W are using stable-baselines3. Pretty good. I like it

1) Integration
   - [x] Animal AI environment runs
      - Use the aniamai.envs.gym.environment wrapper
   - [x] Model weights and internal states can be pulled out
      - Because stable-baselines is open-box
   
2) Dynamic environment generation.
   - [ ] Templated arena manager class
   - [ ] Arena with Ray spaces defined
      - Why? I think we can run grid search in parallel
      - Update: won't do. I want to control the randomization, 
        rather than a grid search.
   - [ ] Tests