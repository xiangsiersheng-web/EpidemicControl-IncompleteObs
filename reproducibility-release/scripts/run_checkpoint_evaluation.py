"""Small fixed-checkpoint high-SPO evaluation; does not train or save models."""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=int,default=2);p.add_argument('--seed',type=int,default=3047);p.add_argument('--device',default='cpu');a=p.parse_args()
 sys.argv=[sys.argv[0]];sys.path.insert(0,str(ROOT))
 from train_gpu import _config_args
 from algorithm.ppo_discrete_gpu import PPO_discrete_gpu
 from environment.uncertain_seir_vector_v4 import EpidemicModel
 from utils.normalization import Normalization
 from uncertainty.obs_imperfect.evaluate_info_rebuild import _evaluate_policy
 args=_config_args();args.device_name=a.device;args.R0='high';args.WINDOW_SIZE=3;args.use_obs_imperfect=False;args.use_rebuild=False;args.use_state_norm=True;args.max_train_steps=48000;args.env_data_dir=str(ROOT/'data')+'/';args.model_idx=18
 agent=PPO_discrete_gpu(args);agent.directory=str(ROOT/'model/gpu/mlp/high/sz_4_maxtrainsteps=48000');agent.load(18)
 norm=Normalization(shape=(args.zone_num,args.local_obs_dim*2),device_name=args.device_name);norm.load(agent.directory,filename='state_norm.pth')
 os.chdir(ROOT/'uncertainty'/'obs_imperfect')
 env=EpidemicModel(args,env_count=a.runs,is_evaluation=True);env.seed(a.seed);info=_evaluate_policy(args,env,agent,state_norm=norm)
 out=ROOT/'reproducibility-release/results/generated';out.mkdir(parents=True,exist_ok=True);path=out/'high_checkpoint_smoke.json';path.write_text(json.dumps({'scenario':'high','paradigm':'SPO','runs':a.runs,'seed':a.seed,'actor':str(Path(agent.directory)/'ppo_actor18.pth'),'mean_C_I':float(info['total_infections'].mean()),'mean_N_test':float(info['total_test_num'].mean())},indent=2));print(path)
if __name__=='__main__':main()
