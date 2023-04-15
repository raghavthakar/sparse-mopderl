## Setup environment
Install Mujoco backend with a bash script all at once, details at [Mujoco's official github](https://github.com/openai/mujoco-py) 
```bash
bash setup.sh
```

With python >= 3.7
```bash
pip install -r requirements.txt
```
## Config environments
All environments' file in ***environments*** folder

## Run MOPDERL
<!-- All bash script for running is in **bash** folder. (Ex: MO-Swimmer-v2 environment):
**Check bash before running**
**Correct example see Swimmer environment**
```bash
bash ./bash/swimmerv2.py
```
Digging into the bash file: -->
```bash
cd MOPDERL/    # <-- cd to MOPDERL directory
python run_mo_pderl.py -env=MO-...-logdir=your/dir -disable_wandb -seed=123 -boundary_only -save_ckpt=0   # <-- Run

#python run_mo_pderl.py -env=MO-... -logdir=your/dir -disable_wandb -seed=987263145 -boundary_only -save_ckpt=0 -checkpoint # <-- Continue running latest run after disconnected

#python run_mo_pderl.py -env=MO-... -logdir=your/dir -disable_wandb -seed=987263145 -boundary_only -save_ckpt=0  -checkpoint -checkpoint_id=10 # <-- Continue running specific run after disconnected
```

Other config please seek into *run_mo_pderl.py* file:  
```python
python run_mo_pderl.py -h
```
<!-- ## Run PGMORL (Skip)
For example, running MO-Swimmer-v2 environment:
```python
cd PGMORL
python scrips/swimmer-v2.py --pgmorl --savedir=../result/PGMORL/MO-Swimmer-v2
``` -->

Results for each run will be at: "*your_save_dir/environment_name/run_***/archive*"


