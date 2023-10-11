# Surface Extraction from Neural Unsigned Distance Fields

### [[Paper](https://arxiv.org/abs/2309.08878)] [[Project Page](https://cong-yi.github.io/projects/dualmeshudf/)]

### Installation

```
git clone https://github.com/cong-yi/DualMesh-UDF.git
cd DualMesh-UDF
conda create -n dmudf python=3.9
conda activate dmudf
pip install .
```
Or you can install it directly by:
```
pip install "git+https://github.com/cong-yi/DualMesh-UDF"
```

### Usage Example
```python
from DualMeshUDF import extract_mesh

# udf_func: function to evaluate UDF values
# udf_grad_func: function to evaluate UDF values and the gradient
mesh_v, mesh_f = extract_mesh(udf_func, udf_grad_func)
```


### An Example for PyTorch
Please note that our implementation is not tied to any specific machine learning framework.

To make it more convenient for PyTorch users, we provide an example with several [checkpoints](https://drive.google.com/drive/folders/12ys47-DjfXC3E-Kt5V2e1DWisynC0rpp?usp=sharing) for testing and demonstration.
Please install [PyTorch](https://pytorch.org/) accordingly and download the checkpoints. Then run the following command:
```
python example/test.py --pretrained [path_to_checkpoint] --mesh_prefix [folder_prefix_for_mesh]
```
The default value for `mesh_prefix` is set to `example/results`. So the output meshes are stored in the folder `example/results`.

For example, given the checkpoints in the folder `example/checkpoints/` and run:
```
python example/test.py --pretrained example/checkpoints/fandisk.pth
```

For other PyTorch-based network, we offer a set of useful tools in `example/neural_utils.py` for reference. With these tools, you can call our method as follows:
```python
mesh_v, mesh_f = extract_mesh_from_udf(net, device)
```

## Citation
If you find our method useful for your research, please cite our paper:

```
@InProceedings{zhang2023dualmeshudf,
    author    = {Zhang, Congyi and
                 Lin, Guying and
                 Yang, Lei and
                 Li, Xin and
                 Komura, Taku and
                 Schaefer, Scott and
                 Keyser, John and
                 Wang, Wenping},
    title     = {Surface Extraction from Neural Unsigned Distance Fields},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {0000-0000}
}
```