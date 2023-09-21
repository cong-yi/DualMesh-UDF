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

### Usage example
```python
from DualMeshUDF import extract_mesh

# udf_func: function to evaluate UDF values
# udf_grad_func: function to evaluate UDF values and the gradient
mesh_v, mesh_f = extract_mesh(udf_func, udf_grad_func)
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