# Start the cloud instance
- If you started the instance please stop it if it is not used anymore.
- Go to https://console.cloud.google.com/compute/instances?project=lidar-sceneflow
- Click on `vm-lidar-sceneflow` and start it

## Initial Setup
### Each user: Add your ssh pub key
- Adding the key to ~/.ssh/authorized_keys does not work as google removes them periodically
- Log into the instance by clicking on ssh -> in browser window
- You will now be logged into the instance with your own user that is derived from your google account. Note this!
- Go to https://console.cloud.google.com/compute/metadata/sshKeys?project=lidar-sceneflow
- Add your ssh public key there and change the "comment" at the end to the username seen previously.
- I guess one could use a different username here but lets stick to the one given by google.
- Now connect using `ssh username@instanceip`

- NOTE: You can login as another use just by specifying its username. But as we are all admins anyway this does not matter. However, please don't.

### One time: Setup the environment
```bash
# For some reason LC_ALL is not set or not supported. Set it to en_US.utf8
export LC_ALL=en_US.utf8
# Create the folder with all the sources
sudo mkdir /opt/fastflow3d
sudo addgroup developers
sudo chgrp -R developers /opt/fastflow3d
# Apply the s bit such that all files are automatically owned by the developers group
# New subdirectories will automaticall inherit the s bit. 
# For existing directories you need to use find with exec: https://askubuntu.com/questions/51951/set-default-group-for-user-when-they-create-new-files
sudo chmod g+s /opt/fastflow3d
sudo chmod -R g+w /opt/fastflow3d
# For every user
sudo usermod -a -G developers <username>
# You need to relog
# Create a ssh key to pull the repo
mkdir /opt/fastflow3d/secrets 
ssh-keygen -f /opt/fastflow3d/secrets/deploykey-rsa -t rsa -b 4096
# Use a password if you like
# Specify this key as deploy key in github or gitlab
cat /opt/fastflow3d/secrets/deploykey-rsa.pub
# Now clone the git repo. Only clone the main branch
GIT_SSH_COMMAND='ssh -i /opt/fastflow3d/secrets/deploykey-rsa' git clone --branch main --single-branch git@github.com:Jabb0/FastFlowNet3D.git /opt/fastflow3d/code

# Create the venv
cd /opt/fastflow3d/code
pip3 install --upgrade pip
pip3 install virtualenv
python3 -m virtualenv venv
# Source the venv and run the requirements install
source venv/bin/activate
pip install -r requirements.txt
```


### For each user
Each user needs to be added to the developer group `sudo usermod -a -G developers <username>`.
Add those two lines to your `~/.bashrc`:
```bash
export LC_ALL=en_US.utf8
export GIT_SSH_COMMAND='ssh -i /opt/fastflow3d/secrets/deploykey-rsa'
```

## Prepare the data
TODO:


## Run Experiments
**Important**: Do NOT change code on the server. Just update it using git pull and run your experiments. Every experiment should have a clean code base and a associated git commit.

The code can be found in `/opt/fastflow3d/code`.

To run an experiment with weights and biases logging you need to specify the weights and biases service user API key.
You will get the key from Felix.

First source the venv `source venv/bin/activate`.

Execute an experiment with:
`python train.py <data_directory> <experiment_name> --wandb_api_key <service_user_api_key>`

You can have multiple runs with the same experiment_name.

Have a look at the other parameters. All parameters for a pytorch lightning trainer are available too (https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#init).