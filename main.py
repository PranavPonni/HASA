import os
import warnings

os.environ.setdefault("WANDB_HTTP_TIMEOUT", "120")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_DISABLE_GIT", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

import torch
import torchvision
import argparse
import importlib
import copy
import itertools
import random
import re
import util
import data_preproc as dp
import wandb
import pdb
from training_speed_utils import configure_reproducibility, get_wandb_entity, wandb_service_settings
from selftouch_offset_utils import input_offset_from_params


class MainExectutor:
    def __init__(
        self,
        mode,
        param_file,
        config,
        sweep_count=None,
        seed=None,
        run_name=None,
        tactile_history=None,
    ):
        params = dp.read_yaml(param_file)

        train_params = params.setdefault("Train", {})
        if seed is None:
            seed = os.environ.get("SELFTOUCH_SEED", train_params.get("seed", 0))
        seed = int(seed)
        deterministic = str(
            os.environ.get("SELFTOUCH_DETERMINISTIC", train_params.get("deterministic", True))
        ).strip().lower() not in {"0", "false", "no", "off"}
        configure_reproducibility(seed, deterministic=deterministic)
        train_params["seed"] = seed
        train_params["deterministic"] = deterministic

        run_name = run_name or os.environ.get("SELFTOUCH_RUN_NAME")
        if run_name:
            run_name = str(run_name).strip()
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
                raise ValueError(
                    "run_name may contain only letters, digits, dot, underscore, and hyphen"
                )
            train_params["run_name"] = run_name

        if tactile_history is None:
            tactile_history = os.environ.get("SELFTOUCH_USE_TACTILE_HISTORY")
        if tactile_history is not None:
            if isinstance(tactile_history, str):
                tactile_history = tactile_history.strip().lower() in {"1", "true", "yes", "on"}
            params.setdefault("Model", {})["use_tactile_history"] = bool(tactile_history)

        params["Experiment"] = {
            "seed": seed,
            "deterministic": deterministic,
            "run_name": run_name or "",
            "use_tactile_history": bool(params.get("Model", {}).get("use_tactile_history", False)),
        }
        
        # params["Model"]["model_name"]=os.path.basename(os.path.dirname(params["Train"]["model_save_path"]))

        self.model_param = params["Model"]
        self.config=config
        self.dataset_param = params["Dataset"]
        self.model_param.setdefault("sequence_length", self.dataset_param.get("sequence_length"))
        self.required_param = params["Required"]
        self.sync_input_offset(params)
        base_model_save_path = self.get_model_save_path(param_file)
        self.model_param["model_name"] = os.path.basename(os.path.dirname(base_model_save_path))
        if run_name:
            self.model_param["model_save_path"] = os.path.join(
                os.path.dirname(base_model_save_path), run_name
            )
            self.dataset_param["param_file_dir"] = os.path.join(
                os.path.dirname(os.path.dirname(param_file)), run_name
            )
            os.makedirs(self.dataset_param["param_file_dir"], exist_ok=True)
            dp.write_yaml(
                params,
                os.path.join(self.dataset_param["param_file_dir"], "parameter.yaml"),
            )
        else:
            self.model_param["model_save_path"] = base_model_save_path
            self.dataset_param["param_file_dir"] = os.path.dirname(param_file)
        print(f"[path] Dataset.data_dir={self.dataset_param.get('data_dir')}")
        self._base_params = copy.deepcopy(params)
        self._base_model_param = copy.deepcopy(self.model_param)
        self._base_dataset_param = copy.deepcopy(self.dataset_param)
        self._base_required_param = copy.deepcopy(self.required_param)
        if mode == "train":
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
            if not bool(params["Sweep"].get("use_wandb_sweep_api", False)):
                self.sweep_run_configs = self.get_local_sweep_run_configs(params)
                self.run_local_sweep_loop(sweep_count)
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

    def sync_input_offset(self, params):
        offset = input_offset_from_params(params.get("Dataset"), params.get("Model"), default=0)
        params.setdefault("Dataset", {})["input_offset"] = offset
        params.setdefault("Model", {})["input_offset"] = offset
        self.dataset_param = params["Dataset"]
        self.model_param = params["Model"]

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
        sweep_params = self.params.get("Sweep", {})
        train_params = self.params.get("Train", {})
        init_kwargs = {"project": sweep_params["project"]}
        entity = get_wandb_entity(self.params.get("Sweep"), self.params.get("Train"), self.params.get("Pretrain"))
        if entity:
            init_kwargs["entity"] = entity
        run_name, model_save_path, param_file_dir = self.reserve_sweep_run_paths(sweep_params)
        self.current_sweep_run_name = run_name
        self.current_sweep_model_save_path = model_save_path
        self.current_sweep_param_file_dir = param_file_dir
        if run_name:
            init_kwargs["name"] = run_name
            init_kwargs["config"] = dict(init_kwargs.get("config", {}), run_name=run_name)
        if sweep_params.get("group"):
            init_kwargs["group"] = str(sweep_params["group"])
        if sweep_params.get("tags"):
            init_kwargs["tags"] = [str(tag) for tag in sweep_params["tags"]]
        if getattr(self, "sweep_run_config", None):
            init_kwargs["config"] = dict(self.sweep_run_config, **init_kwargs.get("config", {}))
            init_kwargs["tags"] = list(init_kwargs.get("tags", [])) + ["fixed-sweep"]
        print(f"[sweep] Output run directory: {run_name}")
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

    def run_local_sweep_loop(self, sweep_count=None):
        configs = getattr(self, "sweep_run_configs", None) or [{}]
        if not configs:
            configs = [{}]

        if sweep_count is None:
            print(
                f"[sweep] Running {len(configs)} local sweep config(s) continuously. "
                "Stop with Ctrl+C."
            )
            run_index = 0
            try:
                while True:
                    config_index = run_index % len(configs)
                    run_index += 1
                    self.reset_sweep_state()
                    self.sweep_run_config = configs[config_index]
                    print(f"[sweep] Starting local sweep run {run_index} config {config_index + 1}/{len(configs)}")
                    completed = self.sweep()
                    if not completed and getattr(self, "_sweep_interrupted", False):
                        break
            except KeyboardInterrupt:
                print("[sweep] Stopped fixed sweep loop with Ctrl+C.")
            return

        sweep_count = int(sweep_count)
        print(f"[sweep] Running {sweep_count} local W&B run(s) from {len(configs)} config(s).")
        for run_index in range(1, sweep_count + 1):
            config_index = (run_index - 1) % len(configs)
            self.reset_sweep_state()
            self.sweep_run_config = configs[config_index]
            print(
                f"[sweep] Starting local sweep run {run_index}/{sweep_count} "
                f"config {config_index + 1}/{len(configs)}"
            )
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

    def get_local_sweep_run_configs(self, params):
        sweep = params.get("Sweep")
        if not isinstance(sweep, dict):
            return [{}]
        tune = sweep.get("tune")
        if not isinstance(tune, dict):
            return [{}]
        parameters = util.get_penultimate_dict(tune)
        names = []
        value_sets = []
        for key, spec in parameters.items():
            if isinstance(spec, dict) and "values" in spec:
                values = spec.get("values")
                if not isinstance(values, (list, tuple)) or not values:
                    raise ValueError(f"Sweep parameter '{key}' has no values")
                names.append(key)
                value_sets.append(list(values))
            elif isinstance(spec, dict) and "value" in spec:
                names.append(key)
                value_sets.append([spec["value"]])
            else:
                raise ValueError(f"Invalid sweep parameter spec for '{key}': {spec}")

        configs = [dict(zip(names, values)) for values in itertools.product(*value_sets)]
        if str(sweep.get("method", "")).lower() == "random":
            random.shuffle(configs)
        return configs or [{}]

    def reserve_sweep_run_paths(self, sweep_params):
        numbered = bool(sweep_params.get("numbered_run_dirs", True))
        model_root = os.path.dirname(self.model_param["model_save_path"])
        param_root = os.path.dirname(os.path.dirname(self.param_file))
        model_name = str(self.model_param.get("model_name") or os.path.basename(model_root))

        if numbered:
            pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)$")
            indices = []
            for root in (model_root, param_root):
                if not os.path.isdir(root):
                    continue
                for name in os.listdir(root):
                    match = pattern.fullmatch(name)
                    if match:
                        indices.append(int(match.group(1)))

            index = max(indices, default=0) + 1
            while True:
                run_name = f"{model_name}_{index:03d}"
                model_save_path = os.path.join(model_root, run_name)
                param_file_dir = os.path.join(param_root, run_name)
                if not os.path.exists(model_save_path) and not os.path.exists(param_file_dir):
                    os.makedirs(model_save_path, exist_ok=False)
                    os.makedirs(param_file_dir, exist_ok=False)
                    return run_name, model_save_path, param_file_dir
                index += 1

        run_name = str(sweep_params.get("run_name") or self.params.get("Train", {}).get("run_name") or model_name)
        model_save_path = os.path.join(model_root, run_name)
        param_file_dir = os.path.join(param_root, run_name)
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(param_file_dir, exist_ok=True)
        return run_name, model_save_path, param_file_dir

    def sweep_config_saver(self,run):
        config = getattr(self, "sweep_run_config", None) or run.config
        self.model_param=util.update_nested_dict(self.model_param,config)
        self.dataset_param=util.update_nested_dict(self.dataset_param,config)
        self.params=util.update_nested_dict(self.params,config)
        self.model_param = dp.localize_legacy_paths(self.model_param)
        self.dataset_param = dp.localize_legacy_paths(self.dataset_param)
        self.params = dp.localize_legacy_paths(self.params)
        run_name = getattr(self, "current_sweep_run_name", run.name)
        self.model_param["model_save_path"] = getattr(
            self,
            "current_sweep_model_save_path",
            os.path.join(os.path.dirname(self.model_param["model_save_path"]), run_name),
        )
        self.dataset_param["param_file_dir"] = getattr(
            self,
            "current_sweep_param_file_dir",
            os.path.join(util.change_dir_name_in_path(os.path.dirname(self.param_file), "parameter_base", run_name)),
        )
        self.params.setdefault("Experiment", {})["run_name"] = run_name
        self.params.setdefault("Train", {})["run_name"] = run_name
        self.params.setdefault("Sweep", {})["run_name"] = run_name
        self.params["Model"] = self.model_param
        self.params["Dataset"] = self.dataset_param
        self.model_param["sequence_length"] = self.dataset_param.get("sequence_length")
        train_params = self.params.get("Train", {})
        use_best_checkpoint = bool(
            train_params.get("save_best_checkpoint", False)
            and int(train_params.get("eval_every", 0) or 0) > 0
        )
        checkpoint_name = (
            "best.pth"
            if use_best_checkpoint
            else f"epoch{int(train_params.get('num_epochs', 1)) - 1}.pth"
        )
        for section in ("Test", "Motion"):
            if section in self.params:
                self.params[section]["model_load_path"] = os.path.join(
                    self.model_param["model_save_path"],
                    checkpoint_name,
                )
        print(f"[path] Sweep Dataset.data_dir={self.dataset_param.get('data_dir')}")
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
    parser.add_argument('-seed', '--seed', type=int, default=None, help='fixed random seed for Python, NumPy, and PyTorch')
    parser.add_argument('-run_name', '--run-name', default=None, help='unique output directory name under model_weight/<variant>')
    parser.add_argument(
        '-tactile_history', '--tactile-history',
        choices=['on', 'off'],
        default=None,
        help='override Model.use_tactile_history for the proprioception-only control',
    )
    args = parser.parse_args()

    tactile_history = None if args.tactile_history is None else args.tactile_history == 'on'
    controller = MainExectutor(
        args.mode,
        args.param_file,
        args.config,
        sweep_count=args.sweep_count,
        seed=args.seed,
        run_name=args.run_name,
        tactile_history=tactile_history,
    )

if __name__ == "__main__":
    main()
