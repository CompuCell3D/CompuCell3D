# Week 0
1. Answer prerequisite Q&A and have a solid understanding of conda and git.
2. Set up a working environment for CompuCell3D.

## Git Self-assessment:
* What is a __fork__?

  A fork is a full and complete clone of the entire repository which will live on one's personal account. It's good for open-source contributions and commonly used when a user does not have write access.
* What is a __branch__?

  A branch is a copy of the progress at a specific point in time that lives inside the same repository. It is primarily used to make changes independently, allowing multiple people to work at the same time without affecting each other until the branches are merged.
* What is a __commit__?

  A commit is like a "Save" button that records the current state of the files that includes a descriptive message and version numbers that allows users to track and go back if needed.
* What is a __pull request__?

  A pull request (PR) is a request made to initiate discussion and review of proposed changes before merging them into another branch of a repository.

### 🔄 How They Work Together (Typical Flow):

🔀 Fork a repo (if you don’t own it).

🌿 Create a branch to isolate your work.

📝 Make commits as you work on changes.

📥 Open a pull request to propose merging your changes.

## Conda Self-assessment:
* What is a __conda environment__?

  A conda environment is a virtual workspace that’s isolated from the system and other environments. It allows you to install or modify packages without causing conflicts or interference with other projects.
* What is a __conda package__?

  A conda package is a compressed file that usually contains prebuilt tools, libraries, and dependencies needed for your environment.
* What is a __conda channel__?

  A conda channel is like a software library or repository where conda packages can be found and downloaded from.
* How can I build a conda environment?

  To build a conda environment, use the command:
<pre>conda create --name [env_name] -c [channel_name] [package]</pre>

__OR__

If you want to recreate a specific environment, you can use a .yml file (called an environment file) that lists all the dependencies and channels.
