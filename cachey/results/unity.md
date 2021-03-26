# Running ray_dummy.py

This is the experience of trying to run AnimalAI with rllib. I'm running on WSL because Ray is experimental for Windows.

Got this error running on Python 3.8:

```
mlagents_envs.exception.UnityTimeOutException: The Unity environment took too long to respond. Make sure that :
	 The environment does not need user interaction to launch
	 The Agents are linked to the appropriate Brains
	 The environment and the Python interface have compatible versions.
```

For me, the problem was that it needs an X Server to be running. My WSL is headless, so I had to install xvfb Github
issue: https://github.com/beyretb/AnimalAI-Olympics/issues/93
Help post: https://forum.unity.com/threads/how-to-run-ml-agents-on-server.994678/
XVFB on WSL: https://www.scivision.dev/xvfb-on-windows-subsystem-for-linux/

It complains of environment file not found because Ray is running in another directory. Ran it with hardcoded absolute
paths.

They recommend running Python 3.6. See
comment: https://github.com/beyretb/AnimalAI-Olympics/issues/28#issuecomment-507062264
