import yaml
import subprocess
import argparse
import os

DEFAULT_CONFIG_FILE = 'sync.yaml'

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def sync_directories(config, key, direction):
    if key not in config:
        print(f"Key {key} not found in configuration.")
        return
    
    key_config = config[key]
    user = key_config['User']
    ip = key_config['IP']
    their_dir = key_config['their_dir']
    our_dir = key_config['our_dir']
    
    if direction in ['out', 'syn']:
        # Construct the rsync command for local to remote sync
        rsync_command = [
            'rsync', '-avz', '--delete', '--no-perms', '--omit-dir-times',
            f"{our_dir}/",                # Source directory (local)
            f"{user}@{ip}:{their_dir}/"   # Destination directory (remote)
        ]
        
        # Execute the rsync command
        result = subprocess.run(rsync_command)
        if result.returncode != 0:
            print(f"Error during rsync from local to remote: {result.stderr}")

    if direction in ['in', 'syn']:
        # Construct the rsync command for remote to local sync
        rsync_command_reverse = [
            'rsync', '-avz', '--delete', '--no-perms', '--omit-dir-times',
            f"{user}@{ip}:{their_dir}/",  # Source directory (remote)
            f"{our_dir}/"                 # Destination directory (local)
        ]
        
        # Execute the reverse rsync command
        result = subprocess.run(rsync_command_reverse)
        if result.returncode != 0:
            print(f"Error during rsync from remote to local: {result.stderr}")

def main():
    parser = argparse.ArgumentParser(description='Synchronize directories using rsync based on YAML configuration.')
    parser.add_argument('key', help='The key of the configuration to use for synchronization')
    parser.add_argument('direction', choices=['in', 'out', 'syn'], help='Direction of synchronization: in, out, or syn')
    parser.add_argument('--config', default=DEFAULT_CONFIG_FILE, help='Path to the YAML configuration file (default: sync.yaml)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found.")
        return
    
    config = load_config(args.config)
    sync_directories(config, args.key, args.direction)

if __name__ == "__main__":
    main()
