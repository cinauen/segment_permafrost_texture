# Installation

The required modules to run this script are specified in the `.yml` files..

The processing steps involving GLCM calculations with the cuml based module `glcm_cuml` (https://github.com/Eve-ning/glcm-cupy) must be run on a
GPU-capable machine (see processing steps marked in [workflow](Module_workflow_2x.png)). All other steps can be run on CPU machines.

The corresponding environmental files are:
* For processes to be run on GPUs: [environment_gpu.yml](environment_gpu.yml)
* For processes to be run on CPUs: [environment_cpu.yml](environment_cpu.yml)


Before installation of the GPU based environment, please check your machine's CUDA version e.g. with:

```bash
nvidia-smi
```

If the CUDA verison is >= 12 the follwoing lines in `environment_gpu.yml`
can be kept however, the cuda-version might need adjusting.

```
  - cuda-version=<cuda-version>
  - cuda-cudart
```

For cuda verisons < 12 the above lines need to be replaced with

```
- cudatoolkit=<cuda-version>
```


The conda environments can then be installed with:

```bash
conda env create -f environment_gpu.yml
```
or

```bash
conda env create -f environment_cpu.yml
```

