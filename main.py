import os
import warnings

os.environ.setdefault("WANDB_HTTP_TIMEOUT", "120")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_DISABLE_GIT", "true")
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

import torch
import torchvision
import argparse
import importlib
import copy
import util
import data_preproc as dp
import wandb
import pdb
from training_speed_utils import get_wandb_entity, wandb_service_settings


class MainExectutor:
    def __init__(self, mode, param_file, config, sweep_count=None):
        params = dp.read_yaml(param_file)
        
        # params["Model"]["model_name"]=os.path.basename(os.path.dirname(params["Train"]["model_save_path"]))

        self.model_param = params["Model"]
        self.config=config
        self.dataset_param = params["Dataset"]
        self.required_param = params["Required"]
        self.model_param["model_save_path"]=self.get_model_save_path(param_file)
        self.model_param["model_name"]=os.path.basename(os.path.dirname(self.model_param["model_save_path"]))
        self.dataset_param["param_file_dir"]=os.path.dirname(param_file)
        self._base_params = copy.deepcopy(params)
        self._base_model_param = copy.deepcopy(self.model_param)
        self._base_dataset_param = copy.deepcopy(self.dataset_param)
        self._base_required_param = copy.deepcopy(self.required_param)
        if mode == "train":
            self.model_param["model_save_path"]=self.get_model_save_path(param_file)
            self.train(params["Train"])
        elif mode == "test":
            self.test(params["Test"])
        elif mode == "motion":
            self.motion(params["Motion"])
        elif mode == "pretrain":
            self.pretrain(params["Pretrain"])
        elif mode == "sweep":
            self.param_file=param_file
            self.params=params
            self.sweep_run_config = self.get_fixed_sweep_run_config(params)
            if self.sweep_run_config is not None and not bool(params["Sweep"].get("use_wandb_sweep_api", False)):
                self.run_fixed_sweep_loop(sweep_count)
                return

            sweep_id=self.get_sweep_config(params)
            agent_kwargs = {"project": params["Sweep"]["project"]}
            entity = get_wandb_entity(params.get("Sweep"), params.get("Train"), params.get("Pretrain"))
            if entity:
                agent_kwargs["entity"] = entity
            if sweep_count is not None:
                agent_kwargs["count"] = sweep_count
                print(f"[sweep] Starting W&B agent for {sweep_count} run(s): {sweep_id}")
            else:
                print(f"[sweep] Starting W&B agent until stopped with Ctrl+C: {sweep_id}")
            wandb.agent(sweep_id, self.sweep, **agent_kwargs)
        else:
            raise ValueError("Invalid mode: {}".format(mode))

    def train(self,train_params):
        
        module = importlib.import_module("model."+self.model_param["model_name"]+".controller")
        controller_instance = getattr(module, self.required_param["controller_name"])
        controller=controller_instance(self.model_param,train_params,self.dataset_param,self.config)
        controller.train_controller()

    def pretrain(self,pretrain_params):
        
        module = importlib.import_module("model."+self.model_param["model_name"]+".controller")
        controller_instance = getattr(module, self.required_param["controller_name"])
        controller=controller_instance(self.model_param,pretrain_params,self.dataset_param,self.config)
        controller.pretrain_controller()

    def sweep(self):
        self._sweep_interrupted = False
        init_kwargs = {"project": self.params["Sweep"]["project"]}
        entity = get_wandb_entity(self.params.get("Sweep"), self.params.get("Train"), self.params.get("Pretrain"))
        if entity:
            init_kwargs["entity"] = entity
        if getattr(self, "sweep_run_config", None):
            init_kwargs["config"] = self.sweep_run_config
            init_kwargs["tags"] = ["fixed-sweep", "selftouch-fcn"]
        run = wandb.init(settings=wandb_service_settings(), **init_kwargs)
        completed = False
        try:
            self.model_param,self.data_param,self.train_param=self.sweep_config_saver(run)
            module = importlib.import_module("model."+self.model_param["model_name"]+".controller")
            controller_instance = getattr(module, self.required_param["controller_name"])
            controller=controller_instance(self.model_param,self.train_param,self.dataset_param,self.config)
            os.makedirs(self.model_param["model_save_path"], exist_ok = True)
            controller.sweep_controller()
            print("Finish training")
            completed = True
        except KeyboardInterrupt:
            print("[sweep] Training interrupted by Ctrl+C; finishing current W&B run.")
            self._sweep_interrupted = True
        except Exception as exc:
            if exc.__class__ is Exception and not str(exc):
                print("[sweep] Training interrupted while CUDA backward was running; finishing current W&B run.")
                self._sweep_interrupted = True
            else:
                raise
        finally:
            wandb.finish()
        return completed

    def reset_sweep_state(self):
        self.params = copy.deepcopy(self._base_params)
        self.model_param = copy.deepcopy(self._base_model_param)
        self.dataset_param = copy.deepcopy(self._base_dataset_param)
        self.required_param = copy.deepcopy(self._base_required_param)
        self.sweep_run_config = self.get_fixed_sweep_run_config(self.params)

    def run_fixed_sweep_loop(self, sweep_count=None):
        if sweep_count is None:
            print("[sweep] Fixed one-combination sweep detected; running online W&B runs continuously. Stop with Ctrl+C.")
            run_index = 0
            try:
                while True:
                    run_index += 1
                    self.reset_sweep_state()
                    print(f"[sweep] Starting fixed sweep run {run_index}")
                    completed = self.sweep()
                    if not completed and getattr(self, "_sweep_interrupted", False):
                        break
            except KeyboardInterrupt:
                print("[sweep] Stopped fixed sweep loop with Ctrl+C.")
            return

        sweep_count = int(sweep_count)
        print(f"[sweep] Fixed one-combination sweep detected; running {sweep_count} online W&B run(s).")
        for run_index in range(1, sweep_count + 1):
            self.reset_sweep_state()
            print(f"[sweep] Starting fixed sweep run {run_index}/{sweep_count}")
            completed = self.sweep()
            if not completed and getattr(self, "_sweep_interrupted", False):
                break
        
    def test(self,test_params):
        module = importlib.import_module("model."+self.model_param["model_name"]+".controller")
        controller_instance = getattr(module, self.required_param["controller_name"])
        controller=controller_instance(self.model_param,test_params,self.dataset_param,self.config)
        controller.test_controller()
        print("test finished")

    def motion(self,motion_params):
        module = importlib.import_module("model."+self.model_param["model_name"]+".controller")
        controller_instance = getattr(module, self.required_param["controller_name"])
        controller=controller_instance(self.model_param,motion_params,self.dataset_param,self.config)
        controller.motion_controller()
        print("motion finished")

    
    def get_model_save_path(self,parameter_file_path):
        file_name = os.path.basename(parameter_file_path)
        model_save_path=util.change_dir_name_in_path(os.path.dirname(parameter_file_path),"parameter","model_weight")
        if file_name.endswith(".yaml"):
            dir_name = file_name[:-5]
        else:
            raise ValueError("Not yaml")
        
        os.makedirs(model_save_path, exist_ok = True)
        return model_save_path
    
    def get_sweep_config(self,params):

        method={"method":params["Sweep"]["method"]}
        metric={"metric":params["Sweep"]["metric"]}
        project={"project":params["Sweep"]["project"]}
      
        hyperparam={"parameters": util.get_penultimate_dict(params["Sweep"]["tune"])}
        sweep_config = {**method,**metric,**project,**hyperparam}
        sweep_kwargs = {"project": params["Sweep"]["project"]}
        entity = get_wandb_entity(params.get("Sweep"), params.get("Train"), params.get("Pretrain"))
        if entity:
            sweep_kwargs["entity"] = entity
        sweep_id = wandb.sweep(sweep_config, **sweep_kwargs)
        return sweep_id

    def get_fixed_sweep_run_config(self, params):
        sweep = params.get("Sweep")
        if not isinstance(sweep, dict):
            return None
        tune = sweep.get("tune")
        if not isinstance(tune, dict):
            return None
        fixed = {}
        parameters = util.get_penultimate_dict(tune)
        for key, spec in parameters.items():
            if isinstance(spec, dict) and "values" in spec:
                values = spec.get("values")
                if not isinstance(values, (list, tuple)) or len(values) != 1:
                    return None
                fixed[key] = values[0]
            elif isinstance(spec, dict) and "value" in spec:
                fixed[key] = spec["value"]
            else:
                return None
        return fixed if fixed else None

    def sweep_config_saver(self,run):
        config = getattr(self, "sweep_run_config", None) or run.config
        self.model_param["model_save_path"]=os.path.join(os.path.dirname(self.model_param["model_save_path"]),run.name)
        self.dataset_param["param_file_dir"]=os.path.join(util.change_dir_name_in_path(os.path.dirname(self.param_file),"parameter_base",run.name))
        self.model_param=util.update_nested_dict(self.model_param,config)
        self.dataset_param=util.update_nested_dict(self.dataset_param,config)
        self.params=util.update_nested_dict(self.params,config)
        os.makedirs(self.dataset_param["param_file_dir"], exist_ok=True)
        dp.write_yaml(self.params,os.path.join(self.dataset_param["param_file_dir"],"parameter.yaml"))
        if "pretrain" in self.config:
            mode_param=self.params["Pretrain"]
        else:
            mode_param=self.params["Train"]

        return self.model_param,self.dataset_param,mode_param

def main():
    parser = argparse.ArgumentParser(
        prog='main.py',
        usage='code for training and testing and data preprocessing',
        description='description',
        epilog='end',
        add_help=True
    )
    parser.add_argument('-mode', choices=['train', 'test', 'motion','sweep', 'pretrain'], help='select mode: train or test or motion')
    parser.add_argument('-param_file', '-param', dest='param_file', help='path to the parameter file')
    parser.add_argument('-config', nargs='+',default=["train"], help='add some description when you want to add config')
    parser.add_argument('-sweep_count', type=int, default=None, help='number of sweep agent runs; omit to run until stopped with Ctrl+C')
    args = parser.parse_args()
    
    controller = MainExectutor(args.mode, args.param_file,args.config, sweep_count=args.sweep_count)

if __name__ == "__main__":
    main()
