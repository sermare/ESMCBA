import numpy as np
import os
import sys
import time
import json
import matplotlib.pyplot as plt
import py3Dmol
from colabdesign.shared.plot import plot_pseudo_3D
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run RFdiffusion with visualization")
parser.add_argument("--name", required=True, help="Output directory name (e.g., EPITOPE_1_16042025)")
parser.add_argument("--pdb", type=str, required=True, help="Input Structure")
parser.add_argument("--contigs", type=str, required=True, help="Contig string (e.g., 9)")
parser.add_argument("--hotspots", required=True, help="Hotspot list")
parser.add_argument("--chains", required=True, help="Chains list")
parser.add_argument("--epitope_chain", required=True, help="Chains of Epitope")
parser.add_argument("--iterations", type=int, required=True, choices=[25, 50, 100, 150, 200], help="Number of RFdiffusion iterations")
parser.add_argument("--num_seqs", type=int, required=True, help="Number of sequences for AlphaFold")

args = parser.parse_args()

pdb = args.pdb

RFDIFFUSION_DIR = "/clusterfs/nilah/sergio/RFdifussion/sokrypton/RFdiffusion"
MODEL_DIR = os.path.join(RFDIFFUSION_DIR, "models")
SCHEDULES_DIR = os.path.join(RFDIFFUSION_DIR, "schedules")
OUTPUT_DIR = os.path.join(f"/clusterfs/nilah/sergio/RFdifussion/", "outputs")
OUTPUT_DIR = os.path.join(OUTPUT_DIR, pdb)
BASE_DIR = "/clusterfs/nilah/sergio/RFdifussion"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Debug sys.path and directories
print("sys.path:", sys.path)
print("BASE_DIR:", BASE_DIR)
print("RFDIFFUSION_DIR:", RFDIFFUSION_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("SCHEDULES_DIR:", SCHEDULES_DIR)
print("Exists BASE_DIR:", os.path.exists(BASE_DIR))
print("Exists RFDIFFUSION_DIR:", os.path.exists(RFDIFFUSION_DIR))
print("Exists MODEL_DIR:", os.path.exists(MODEL_DIR))
print("Exists SCHEDULES_DIR:", os.path.exists(SCHEDULES_DIR))

# Verify ananas binary
ANANAS_PATH = os.path.join(BASE_DIR, "ananas")
if not os.path.exists(ANANAS_PATH):
    print("Downloading ananas...")
    os.system(f"wget -qnc https://files.ipd.uw.edu/krypton/ananas -O {ANANAS_PATH}")
    os.system(f"chmod +x {ANANAS_PATH}")

# Verify model files
required_models = ["Base_ckpt.pt", "Complex_base_ckpt.pt"]
for model in required_models:
    model_path = os.path.join(MODEL_DIR, model)
    print(f"Checking model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model} not found at {model_path}")

# Verify schedules directory
if not os.path.exists(SCHEDULES_DIR):
    raise FileNotFoundError(f"Schedules directory not found at {SCHEDULES_DIR}")

# Add RFdiffusion to sys.path
if RFDIFFUSION_DIR not in sys.path:
    os.environ["DGLBACKEND"] = "pytorch"
    sys.path.append(RFDIFFUSION_DIR)
    print(f"Added {RFDIFFUSION_DIR} to sys.path")

if os.path.join(RFDIFFUSION_DIR, "rfdiffusion") not in sys.path:
    sys.path.append(os.path.join(RFDIFFUSION_DIR, "rfdiffusion"))


# Import RFdiffusion utilities
try:
    from inference.utils import parse_pdb
    from colabdesign.rf.utils import get_ca, fix_contigs, fix_partial_contigs, fix_pdb, sym_it
    from colabdesign.shared.protein import pdb_to_string
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print("Ensure ColabDesign is installed: pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
    print("Verify RFdiffusion directory at", RFDIFFUSION_DIR)
    print("Contents of RFDIFFUSION_DIR:", os.listdir(RFDIFFUSION_DIR))
    raise

def get_pdb(pdb_code=None):
    """Retrieve or validate PDB file."""
    if pdb_code is None or pdb_code == "":
        raise ValueError("No PDB code provided.")
    elif os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        pdb_file = os.path.join(BASE_DIR, f"{pdb_code}.pdb1")
        if not os.path.exists(pdb_file):
            os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.pdb1.gz -O {pdb_file}.gz")
            os.system(f"gunzip {pdb_file}.gz")
        return pdb_file
    else:
        pdb_file = os.path.join(BASE_DIR, f"AF-{pdb_code}-F1-model_v3.pdb")
        if not os.path.exists(pdb_file):
            os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb -O {pdb_file}")
        return pdb_file

def run_ananas(pdb_str, path, sym=None):
    """Run AnAnaS for symmetry detection."""
    os.makedirs(os.path.join(OUTPUT_DIR, path), exist_ok=True)
    pdb_filename = os.path.join(OUTPUT_DIR, path, "ananas_input.pdb")
    out_filename = os.path.join(OUTPUT_DIR, path, "ananas.json")
    with open(pdb_filename, "w") as handle:
        handle.write(pdb_str)

    cmd = f"{ANANAS_PATH} {pdb_filename} -u -j {out_filename}"
    if sym:
        cmd += f" {sym}"
    os.system(cmd)

    try:
        out = json.loads(open(out_filename, "r").read())
        results, AU = out[0], out[-1]["AU"]
        group = AU["group"]
        chains = AU["chain names"]
        rmsd = results["Average_RMSD"]
        print(f"AnAnaS detected {group} symmetry at RMSD:{rmsd:.3}")

        C = np.array(results['transforms'][0]['CENTER'])
        A = [np.array(t["AXIS"]) for t in results['transforms']]

        new_lines = []
        for line in pdb_str.split("\n"):
            if line.startswith("ATOM"):
                chain = line[21:22]
                if chain in chains:
                    x = np.array([float(line[i:(i+8)]) for i in [30, 38, 46]])
                    if group[0] == "c":
                        x = sym_it(x, C, A[0])
                    if group[0] == "d":
                        x = sym_it(x, C, A[1], A[0])
                    coord_str = "".join(["{:8.3f}".format(a) for a in x])
                    new_lines.append(line[:30] + coord_str + line[54:])
            else:
                new_lines.append(line)
        return results, "\n".join(new_lines)
    except:
        return None, pdb_str

def run(command, steps, num_designs=1, visual="none"):
    """Execute RFdiffusion command and monitor progress."""
    def run_command_and_get_pid(command):
        pid_file = os.path.join(BASE_DIR, "pid")
        os.system(f'{command} > {BASE_DIR}/run.log 2>&1 & echo $! > {pid_file}')
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        os.remove(pid_file)
        return pid

    def is_process_running(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    print(f"Running command: {command}")
    pid = run_command_and_get_pid(command)
    try:
        for _ in range(num_designs):
            for n in range(steps):
                wait = True
                while wait:
                    time.sleep(1)
                    pdb_file = os.path.join(BASE_DIR, f"{n}.pdb")
                    if os.path.exists(pdb_file):
                        with open(pdb_file, "r") as f:
                            pdb_str = f.read()
                        if pdb_str.strip().endswith("TER"):
                            wait = False
                        elif not is_process_running(pid):
                            print("Process failed.")
                            return
                    elif not is_process_running(pid):
                        print("Process terminated unexpectedly.")
                        return

                if visual != "none":
                    if visual == "image":
                        xyz, bfact = get_ca(pdb_file, get_bfact=True)
                        fig = plt.figure(figsize=(6, 6), dpi=100)
                        ax = fig.add_subplot(111)
                        ax.set_xticks([]); ax.set_yticks([])
                        plot_pseudo_3D(xyz, c=bfact, cmin=0.5, cmax=0.9, ax=ax)
                        plt.savefig(os.path.join(OUTPUT_DIR, f"step_{n}.png"))
                        plt.close()
                    elif visual == "interactive":
                        view = py3Dmol.view()
                        view.addModel(pdb_str, 'pdb')
                        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 0.5, 'max': 0.9}}})
                        view.zoomTo()
                        view.save_html(os.path.join(OUTPUT_DIR, f"step_{n}.html"))

                if os.path.exists(pdb_file):
                    os.remove(pdb_file)

        while is_process_running(pid):
            time.sleep(1)

    except KeyboardInterrupt:
        os.system(f"kill -TERM {pid}")
        print("Process stopped.")

def run_diffusion(contigs, path, pdb=None, iterations=50,
                  symmetry="none", order=1, hotspot=None,
                  chains=None, add_potential=False,
                  num_designs=1, visual="none"):
    """Run RFdiffusion inference."""
    full_path = os.path.join(OUTPUT_DIR, path)
    os.makedirs(full_path, exist_ok=True)
    opts = [f"inference.output_prefix={full_path}",
            f"inference.num_designs={num_designs}"]

    if chains == "":
        chains = None

    # Determine symmetry type
    if symmetry in ["auto", "cyclic", "dihedral"]:
        if symmetry == "auto":
            sym, copies = None, 1
        else:
            sym, copies = {"cyclic": (f"c{order}", order),
                          "dihedral": (f"d{order}", order*2)}[symmetry]
    else:
        symmetry = None
        sym, copies = None, 1

    # Determine mode
    contigs = contigs.replace(",", " ").replace(":", " ").split()
    is_fixed, is_free = False, False
    fixed_chains = []
    for contig in contigs:
        for x in contig.split("/"):
            a = x.split("-")[0]
            if a[0].isalpha():
                is_fixed = True
                if a[0] not in fixed_chains:
                    fixed_chains.append(a[0])
            if a.isnumeric():
                is_free = True
    if len(contigs) == 0 or not is_free:
        mode = "partial"
    elif is_fixed:
        mode = "fixed"
    else:
        mode = "free"

    # Fix input contigs
    if mode in ["partial", "fixed"]:
        pdb_str = pdb_to_string(get_pdb(pdb), chains=chains)
        if symmetry == "auto":
            a, pdb_str = run_ananas(pdb_str, path)
            if a is None:
                print("ERROR: no symmetry detected")
                symmetry = None
                sym, copies = None, 1
            else:
                if a["group"][0] == "c":
                    symmetry = "cyclic"
                    sym, copies = a["group"], int(a["group"][1:])
                elif a["group"][0] == "d":
                    symmetry = "dihedral"
                    sym, copies = a["group"], 2 * int(a["group"][1:])
                else:
                    print(f"ERROR: detected symmetry ({a['group']}) not supported")
                    symmetry = None
                    sym, copies = None, 1

        elif mode == "fixed":
            pdb_str = pdb_to_string(pdb_str, chains=fixed_chains)

        pdb_filename = os.path.join(full_path, "input.pdb")
        with open(pdb_filename, "w") as handle:
            handle.write(pdb_str)

        parsed_pdb = parse_pdb(pdb_filename)
        opts.append(f"inference.input_pdb={pdb_filename}")
        if mode == "partial":
            iterations = int(80 * (iterations / 200))
            opts.append(f"diffuser.partial_T={iterations}")
            contigs = fix_partial_contigs(contigs, parsed_pdb)
        else:
            opts.append(f"diffuser.T={iterations}")
            contigs = fix_contigs(contigs, parsed_pdb)
    else:
        opts.append(f"diffuser.T={iterations}")
        parsed_pdb = None
        contigs = fix_contigs(contigs, parsed_pdb)

    if hotspot and hotspot != "":
        opts.append(f"ppi.hotspot_res=[{hotspot}]")

    # Setup symmetry
    if sym is not None:
        sym_opts = ["--config-name symmetry", f"inference.symmetry={sym}"]
        if add_potential:
            sym_opts += ["'potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"]'",
                        "potentials.olig_intra_all=True", "potentials.olig_inter_all=True",
                        "potentials.guide_scale=2", "potentials.guide_decay=quadratic"]
        opts = sym_opts + opts
        contigs = sum([contigs] * copies, [])

    opts.append(f"'contigmap.contigs=[{' '.join(contigs)}]'")
    opts.append("inference.dump_pdb=True")
    opts.append(f"inference.dump_pdb_path={full_path}")

    print("Mode:", mode)
    print("Output:", full_path)
    print("Contigs:", contigs)

    opts_str = " ".join(opts)
    cmd = f"python {RFDIFFUSION_DIR}/run_inference.py {opts_str}"
    print("Command:", cmd)

    # Run
    run(cmd, iterations, num_designs, visual=visual)

    # Fix PDBs
    for n in range(num_designs):
        pdbs = [f"{full_path}/traj/{path}_{n}_pX0_traj.pdb",
                f"{full_path}/traj/{path}_{n}_Xt-1_traj.pdb",
                f"{full_path}_{n}.pdb"]
        for pdb in pdbs:
            if os.path.exists(pdb):
                with open(pdb, "r") as handle:
                    pdb_str = handle.read()
                with open(pdb, "w") as handle:
                    handle.write(fix_pdb(pdb_str, contigs))

    return contigs, copies

import os
import subprocess

BASE_DIR = "/clusterfs/nilah/sergio/RFdifussion"
PARAM_DIR = os.path.join(BASE_DIR, "params")

# Create param dir if not exists
os.makedirs(PARAM_DIR, exist_ok=True)

# Download AlphaFold params only if needed
done_file = os.path.join(PARAM_DIR, "done.txt")
if not os.path.isfile(done_file):
    print("Downloading AlphaFold parameters...")
    alphafold_tar = os.path.join(PARAM_DIR, "alphafold_params_2022-12-06.tar")

    subprocess.run([
        "wget",
        "-O", alphafold_tar,
        "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    ], check=True)

    print("Extracting AlphaFold parameters...")
    subprocess.run(["tar", "-xf", alphafold_tar, "-C", PARAM_DIR], check=True)

    with open(done_file, "w") as f:
        f.write("Done.")

def visualize_epitope_mhc(pdb_path, output_png, epitope_chain="D"):
    """
    Visualize the epitope-MHC complex from a PDB file and save as PNG.
    
    Args:
        pdb_path (str): Path to the RFdiffusion output PDB (e.g., RUN2_0.pdb).
        output_png (str): Path to save the PNG image.
        epitope_chain (str): Chain ID of the epitope (assumed to be the designed peptide).
    """
    # Read PDB file
    with open(pdb_path, "r") as f:
        pdb_str = f.read()

    # Initialize py3Dmol view
    view = py3Dmol.view(width=800, height=600)

    # Add the PDB model
    view.addModel(pdb_str, "pdb")

    # Style for MHC (chains A, B, C)
    view.setStyle({"chain": {"A", "B", "C"}}, 
                  {"cartoon": {"color": "lightblue", "opacity": 0.8}})

    # Style for epitope (assumed to be chain D or the designed peptide)
    view.setStyle({"chain": epitope_chain}, 
                  {"cartoon": {"color": "red", "thickness": 1.0}, 
                   "stick": {"color": "red"}})

    # Zoom and center
    view.zoomTo()
    view.setBackgroundColor("white")

    # Save as PNG
    view.render()
    view.png(output_png) #, width=800, height=600)
    print(f"Saved PNG to {output_png}")

    # Alternative: Pseudo-3D plot with matplotlib
    xyz, bfact = get_ca(pdb_path, get_bfact=True)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xticks([]); ax.set_yticks([])
    plot_pseudo_3D(xyz, c=bfact, cmin=0.5, cmax=0.9, ax=ax)
    plt.savefig(output_png.replace(".png", "_pseudo3D.png"))
    plt.close()
    print(f"Saved pseudo-3D PNG to {output_png.replace('.png', '_pseudo3D.png')}")

# Set parameters from arguments
name = args.name
contigs = args.contigs
# contigs = "A1-274/0 B1-100/0 " + contigs
iterations = args.iterations
num_seqs = args.num_seqs

hotspot = args.hotspots
chains = args.chains
rm_aa = args.epitope_chain

# Fixed parameters
pdb = f"/clusterfs/nilah/sergio/RFdifussion/structures/pdb_files/{pdb.lower()}.pdb"

hotspot = "A5,A9,A33,A45,A59,A66,A67,A69,A70,A73,A76,A77,A80,A81,A84,A95,A97,A114,A123,A124,A142,A147,A152,A159" #ONLY HIGH CONTACT

## Added this 20th May 2025 for controls of no hotspots
hotspot = "A1"

num_designs = 1
# contigs = "A1-274/0 B1-100/0 C1-9"
visual = "none"
symmetry = "none"
order = 1
# chains = "A,B,C"
add_potential = False

#Print Parameters for RFdiffusion
print("Parameters for RFdiffusion:")   
print(f"Name: {name}")
print(f"PDB: {pdb}")
print(f"Contigs: {contigs}")
print(f"Iterations: {iterations}")
print(f"Symmetry: {symmetry}")
print(f"Hotspot: {hotspot}")
print(f"Chains: {chains}")
print(f"Order: {order}")
print(f"Add Potential: {add_potential}")
print(f"Num Designs: {num_designs}")
print(f"Visual: {visual}")
print(f"Num Seqs: {num_seqs}")
print(f"Epitope Chain: {rm_aa}")
print(f"PDB DIRECTORY: {pdb}")

## Parameters for AF
initial_guess = False
num_recycles = 6
use_multimer = True
rm_aa = "C"
mpnn_sampling_temp = 0.5

# determine where to save
path = name
while os.path.exists(f"outputs/{name}_0.pdb"):
  path = name + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

OUTPUT_DIR = os.path.join(OUTPUT_DIR, path)

flags = {"contigs":contigs,
         "pdb":pdb,
         "order":order,
         "iterations":iterations,
         "symmetry":symmetry,
         "hotspot":hotspot,
         "path":path,
         "chains":chains,
         "add_potential":add_potential,
         "num_designs":num_designs,
         "visual":visual}

for k,v in flags.items():
  if isinstance(v,str):
    flags[k] = v.replace("'","").replace('"','')
# %%time

print("\n \n \n Running RFdiffusion:") 

contigs, copies = run_diffusion(**flags)

import os, time, subprocess

# Wait for AlphaFold params
max_wait = 300
elapsed = 0

# Absolute path setup
output_dir = f'{OUTPUT_DIR}/'
contigs_str = ":".join(contigs)

opts = [
    f"--pdb={output_dir}{name}_0.pdb",
    f"--loc={output_dir}",
    f"--contig={contigs_str}",
    f"--copies={copies}",
    f"--num_seqs={num_seqs}",
    f"--num_recycles={num_recycles}",
    f"--rm_aa={rm_aa}",
    f"--mpnn_sampling_temp={mpnn_sampling_temp}",
    f"--num_designs={num_designs}",
]
if initial_guess: opts.append("--initial_guess")
if use_multimer: opts.append("--use_multimer")
# opts.append("--num_models=1")

# Set AlphaFold weights path for ColabDesign
os.environ["COLABDESIGN_AF_DIR"] = "/clusterfs/nilah/sergio/RFdifussion/params"

print("\n \n \n Running AlphaFold:") 

subprocess.run(["python", "/clusterfs/nilah/sergio/miniconda3/envs/SE3nv/lib/python3.9/site-packages/colabdesign/rf/designability_test.py"] + opts, check=True)

#cat /clusterfs/nilah/sergio/RFdifussion/run.log

# Visualization
# from Bio.PDB import PDBParser, Superimposer, PDBIO

# alt_pdb_input_path = "/clusterfs/nilah/sergio/RFdifussion/RFdiffusion/structures/7RTD.pdb"

# # Check MHC file
# if os.path.exists(alt_pdb_input_path):
#     pdb_input_path = alt_pdb_input_path
# elif not os.path.exists(pdb_input_path):
#     # Try downloading using get_pdb logic
#     pdb_file = os.path.join(BASE_DIR, f"{pdb_input}.pdb1")
#     if not os.path.exists(pdb_file):
#         os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_input}.pdb1.gz -O {pdb_file}.gz")
#         os.system(f"gunzip {pdb_file}.gz")
#     pdb_input_path = pdb_file
# if not os.path.exists(pdb_input_path):
#     raise FileNotFoundError(f"MHC PDB file not found at {pdb_input_path} or alternatives")

# # Ensure OUTPUT_DIR exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def align_epitope_to_mhc(epitope_pdb, mhc_pdb, epitope_chain="A", ref_chain="C"):
#     """
#     Align epitope to the reference peptide in MHC (chain C).
    
#     Args:
#         epitope_pdb (str): Path to epitope PDB (e.g., EPITOPE_TRIAL8/0.pdb).
#         mhc_pdb (str): Path to MHC PDB (7rtd.pdb).
#         epitope_chain (str): Chain ID of epitope in epitope_pdb.
#         ref_chain (str): Chain ID of reference peptide in mhc_pdb.
    
#     Returns:
#         str: Path to aligned epitope PDB.
#     """
#     parser = PDBParser(QUIET=True)
#     try:
#         mhc_struct = parser.get_structure("mhc", mhc_pdb)
#         epitope_struct = parser.get_structure("epitope", epitope_pdb)
#     except Exception as e:
#         print(f"Error parsing PDB files: {e}")
#         return epitope_pdb
    
#     # Get C-alpha atoms
#     try:
#         ref_atoms = [a for a in mhc_struct[0][ref_chain].get_atoms() if a.name == "CA"]
#         moving_atoms = [a for a in epitope_struct[0][epitope_chain].get_atoms() if a.name == "CA"]
#     except KeyError as e:
#         print(f"Chain {ref_chain} not found in MHC or {epitope_chain} in epitope: {e}")
#         return epitope_pdb
    
#     # Ensure 9 residues
#     if len(ref_atoms) < 9 or len(moving_atoms) != 9:
#         print(f"Warning: Reference chain {ref_chain} has {len(ref_atoms)} CA atoms, epitope has {len(moving_atoms)}")
#         return epitope_pdb
    
#     # Align
#     sup = Superimposer()
#     sup.set_atoms(ref_atoms[:9], moving_atoms[:9])
#     sup.apply(epitope_struct[0])
    
#     # Save aligned epitope
#     aligned_pdb = epitope_pdb.replace(".pdb", "_aligned.pdb")
#     io = PDBIO()
#     io.set_structure(epitope_struct)
#     io.save(aligned_pdb)
#     return aligned_pdb

# def combine_epitope_mhc(epitope_pdb, mhc_pdb, epitope_chain="D"):
#     """
#     Combine epitope and MHC structures into a single PDB string.
    
#     Args:
#         epitope_pdb (str): Path to epitope PDB.
#         mhc_pdb (str): Path to MHC PDB.
#         epitope_chain (str): New chain ID for epitope.
    
#     Returns:
#         str: Combined PDB string.
#     """
#     try:
#         mhc_str = pdb_to_string(mhc_pdb, chains=["A", "B", "C"])
#     except Exception as e:
#         raise RuntimeError(f"Failed to load MHC PDB {mhc_pdb}: {e}")
    
#     if not os.path.exists(epitope_pdb):
#         raise FileNotFoundError(f"Epitope PDB {epitope_pdb} not found")

#     with open(epitope_pdb, "r") as f:
#         epitope_lines = f.readlines()

#     epitope_str = ""
#     for line in epitope_lines:
#         if line.startswith("ATOM"):
#             line = line[:21] + epitope_chain + line[22:]
#             epitope_str += line
#         elif line.startswith("TER") or line.startswith("END"):
#             epitope_str += f"TER   {epitope_chain}\n"

#     return mhc_str + "TER\n" + epitope_str + "END\n"

# def visualize_epitope_mhc(epitope_pdb, mhc_pdb, output_base, epitope_chain="D"):
#     """
#     Visualize the epitope-MHC complex and save as PNG using PyMOL.
    
#     Args:
#         epitope_pdb (str): Path to epitope PDB.
#         mhc_pdb (str): Path to MHC PDB.
#         output_base (str): Base path for output files.
#         epitope_chain (str): Chain ID for the epitope.
#     """
#     # Align epitope
#     aligned_epitope_pdb = align_epitope_to_mhc(epitope_pdb, mhc_pdb, epitope_chain="A", ref_chain="C")
    
#     # Combine structures
#     combined_pdb_str = combine_epitope_mhc(aligned_epitope_pdb, mhc_pdb, epitope_chain)
#     combined_pdb_file = output_base + "_combined.pdb"
#     with open(combined_pdb_file, "w") as f:
#         f.write(combined_pdb_str)

#     # Visualize with PyMOL
#     try:
#         import pymol
#         from pymol import cmd
#         cmd.reinitialize()
#         cmd.load(combined_pdb_file, "complex")
        
#         # Style MHC
#         cmd.set("cartoon_color", "lightblue", "complex and chain A+B+C")
#         cmd.set("cartoon_transparency", 0.2, "complex and chain A+B+C")
        
#         # Style epitope
#         cmd.set("cartoon_color", "red", f"complex and chain {epitope_chain}")
#         cmd.show("sticks", f"complex and chain {epitope_chain}")
        
#         # Style hotspots
#         # hotspot_residues = [7, 9, 59, 63, 66, 70, 99, 159, 167]
#         # cmd.show("spheres", f"complex and chain A and resi {'+'.join(map(str, hotspot_residues))}")
#         # cmd.set("sphere_color", "yellow", f"complex and chain A and resi {'+'.join(map(str, hotspot_residues))}")
#         # cmd.set("sphere_transparency", 0.3)
        
#                 # Orient the camera to look top-down onto the epitope (Chain D)
#         cmd.orient(f"complex and chain {epitope_chain}")
#         # cmd.turn("x", 90) 
#         # cmd.turn("z", 0)
#         # cmd.turn("y", 45)  # or "y", "z" — adjust depending on your structure
#         # cmd.zoom(f"complex and chain {epitope_chain}", 5)  # Zoom tighter on epitope
#         cmd.set_view((0.5800932049751282, 0.019943242892622948, -0.81430584192276, 0.7175763845443726, 0.4605588912963867, 0.5224648714065552, 0.38545551896095276, -0.8874051570892334, 0.2528563439846039, 0.0, 0.0, -109.06678771972656, 13.777585983276367, 12.794111251831055, 66.28702545166016, 85.98908996582031, 132.1444854736328, -20.0))


#         # Zoom and save
#         # cmd.zoom()
#         png_path = output_base + ".png"
#         cmd.png(png_path, width=800, height=600)
#         # print(f"Saved PNG to {png_path}")
        
#         # Clean up
#         cmd.delete("all")
#         # os.remove(combined_pdb_file)
#         if aligned_epitope_pdb != epitope_pdb:
#             os.remove(aligned_epitope_pdb)
#     except Exception as e:
#         print(f"Error visualizing with PyMOL: {e}")
#         return

#     # Pseudo-3D plot
#     try:
#         xyz, bfact = get_ca(epitope_pdb, get_bfact=True)
#         fig = plt.figure(figsize=(6, 6), dpi=100)
#         ax = fig.add_subplot(111)
#         ax.set_xticks([]); ax.set_yticks([])
#         plot_pseudo_3D(xyz, c=bfact, cmin=0.5, cmax=0.9, ax=ax)
#         pseudo_png_path = output_base + "_pseudo3D.png"
#         plt.savefig(pseudo_png_path)
#         plt.close()
#         # print(f"Saved pseudo-3D plot to {pseudo_png_path}")
#     except Exception as e:
#         print(f"Error creating pseudo-3D plot: {e}")

# for i in range(0, iterations):
#     epitope_pdb = os.path.join(OUTPUT_DIR, f"{name}/{i}.pdb")
#     output_base = os.path.join(OUTPUT_DIR, f"step_{i}_epitope_mhc")
#     if os.path.exists(epitope_pdb):
#         try:
#             visualize_epitope_mhc(epitope_pdb, pdb_input_path, output_base, epitope_chain="D")
#         except Exception as e:
#             print(f"Failed to visualize {epitope_pdb}: {e}")
#     else:
#         print(f"Missing epitope: {epitope_pdb}")