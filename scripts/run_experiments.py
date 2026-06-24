import os
import sys
import time
import subprocess
import re
import shutil

def update_config(use_curriculum: bool, use_topo: bool):
    config_path = "scripts/train_unetpp.py"
    with open(config_path, "r") as f:
        content = f.read()

    # Replace USE_CURRICULUM
    content = re.sub(
        r"(USE_CURRICULUM\s*=\s*)(True|False)",
        f"\\g<1>{use_curriculum}",
        content
    )
    # Replace USE_TOPO_LOSS
    content = re.sub(
        r"(USE_TOPO_LOSS\s*=\s*)(True|False)",
        f"\\g<1>{use_topo}",
        content
    )

    with open(config_path, "w") as f:
        f.write(content)
    print(f"[CONFIG] Updated: USE_CURRICULUM={use_curriculum}, USE_TOPO_LOSS={use_topo}")

def wait_for_process(name_pattern):
    print(f"[WAIT] Waiting for existing processes matching '{name_pattern}' to finish...")
    while True:
        try:
            # Check if process is running
            output = subprocess.check_output(["pgrep", "-fl", name_pattern]).decode()
            # Filter out this script itself
            lines = [line for line in output.split('\n') if line and "run_experiments.py" not in line]
            if not lines:
                break
            print(f"[WAIT] Active processes: {lines}. Retrying in 15s...")
        except subprocess.CalledProcessError:
            # pgrep returns non-zero when no processes match
            break
        time.sleep(15)
    print("[WAIT] No matching processes found. Proceeding.")

def copy_results(exp_name):
    dest_dir = f"results/ablation_results/{exp_name}"
    os.makedirs(f"{dest_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{dest_dir}/evaluation", exist_ok=True)

    src_ckpt = "results/checkpoints_unetpp"
    src_eval = "results/evaluation_results_unetpp"

    if os.path.exists(src_ckpt):
        for f in os.listdir(src_ckpt):
            shutil.copy(os.path.join(src_ckpt, f), os.path.join(dest_dir, "checkpoints", f))
    if os.path.exists(src_eval):
        for f in os.listdir(src_eval):
            shutil.copy(os.path.join(src_eval, f), os.path.join(dest_dir, "evaluation", f))
    print(f"[COPY] Copied results for {exp_name} to {dest_dir}")

def run_command(cmd):
    print(f"[RUN] Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            sys.stdout.write(output.decode())
            sys.stdout.flush()
    rc = process.poll()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {cmd}")
    print(f"[RUN] Completed successfully: {cmd}")

def main():
    # 1. Wait for any currently running train_unetpp.py or evaluate_unetpp.py to finish
    wait_for_process("train_unetpp")
    wait_for_process("evaluate_unetpp")

    # Give a few seconds for files to finish writing
    time.sleep(5)

    # 2. Copy current results (which is Experiment A, since we updated config to USE_CURRICULUM=False, USE_TOPO_LOSS=False)
    print("\n--- Processing Experiment A (Baseline) results ---")
    copy_results("exp_a_baseline")

    # 3. Experiment C: USE_CURRICULUM = False, USE_TOPO_LOSS = True
    print("\n--- Starting Experiment C (+Topo only) ---")
    update_config(use_curriculum=False, use_topo=True)
    try:
        run_command(".venv/bin/python scripts/train_unetpp.py")
        run_command(".venv/bin/python scripts/evaluate_unetpp.py")
        copy_results("exp_c_topo")
    except Exception as e:
        print(f"[ERROR] Experiment C failed: {e}")
        return

    # 4. Experiment D: USE_CURRICULUM = True, USE_TOPO_LOSS = True (Ours)
    print("\n--- Starting Experiment D (Full Method - Ours) ---")
    update_config(use_curriculum=True, use_topo=True)
    try:
        run_command(".venv/bin/python scripts/train_unetpp.py")
        run_command(".venv/bin/python scripts/evaluate_unetpp.py")
        copy_results("exp_d_ours")
    except Exception as e:
        print(f"[ERROR] Experiment D failed: {e}")
        return

    print("\n==================================================")
    print("ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
