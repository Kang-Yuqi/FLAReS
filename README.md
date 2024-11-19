# FLAReS
**FLAReS** (**F**ull-sky **L**ensing with **A**daptive **Re**solution **S**hells) provides a python package for full-sky weak lensing simulations with non-Gaussianity analysis. 

## Characteristics
* Construct spherical lens shell from Nbody boxes.
* Efficient Nboy box arrangement for lens shells in differnt redshift.
* Measure non-Gaussianity parameters (Skewness & Kurtosis) of lensing maps either locally or globally.

## Installation

To install FLAReS, navigate to the directory where you downloaded the package and install it using `pip`:

```bash
cd FLAReS
pip install .
```

## Example 
* A quarter cross-section view (to scale) of one of the example lens shell arrangements for CMB lensing simulation. The shells are distinguished by different colors, and the grid represents the scaled N-body box
  
  <img src="https://github.com/user-attachments/assets/67d305e7-ece6-4244-b6fb-4dd2b4528633" alt="nbody_arrange_redshift" width="300">

* N-body box stacking strategies for constructing one of the lensing shells.
  
  <img src="https://github.com/user-attachments/assets/258510e2-5866-4412-81fd-edb8144a142f" alt="stacking_box_shell_combine" width="500">

## FLAReS Workflow

This section provides a step-by-step guide for generating, running, and processing N-body simulations, shell construction, and lensing calculations using FLAReS.

### 1. Generate Parameter File

The parameter file `parameter.ini` contains all the required simulation and analysis parameters. Ensure this file is prepared before proceeding to the next steps. 
Examples for the parameter file are provided in the `FLAReS/param` folder.

### 2. Generate Shell Arrangement and Gadget-4 Configuration

Run the following command to generate the necessary files, including shell arrangement and N-body configuration files for `Gadget-4`:

```bash
python gen_Nbody_run.py -p parameter.ini
```

#### **Output**
- Shell arrangement files for your simulation.
- N-body configuration files required to run `Gadget-4`.

### 3. Run Gadget-4 Simulations

#### **Directory for Configuration Files**
The generated N-body configuration files and related data will be stored in:
`/FLAReS/projects/<project-name>/<Nbody_folder-name>`

#### **Example PBS Scripts for Cluster**
If running on a PBS cluster, example job scripts are available at:
`/FLAReS/projects/<project-name>/cluster`

Submit the PBS job to the cluster system:

`qsub example_script.pbs`

### 4. Constructing the Shell

To construct the lensing shell, run:

```bash
python shell_run.py -p parameter.ini
```

#### **Output**
- Lensing shell maps will be generated in the following directory:
  `/FLAReS/projects/<project-name>/shells`

### 5. Correcting Shell Thickness

Generate shell thickness corrections to account for the effects of finite shell thickness:

```bash
python shell_thickness_correction.py -p parameter.ini
```

#### **Output**
The correction data will be stored in spherical harmonic space within the corresponding shell directory.

### 6. Lensing Process

Perform lensing calculations with either post-Born corrections or the Born approximation.

#### **Lensing with Post-Born Corrections**

```bash
python lensing_post-Born_run.py -p parameter.ini
```

#### **Lensing with Born Approximation**

```bash
python lensing_post-Born_run.py -p parameter.ini
```


## Developer
Yuqi Kang
