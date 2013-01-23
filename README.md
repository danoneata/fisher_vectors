Fisher vectors
==============

A brief description of the pipeline.

The main stages are the following:
1. Extract descriptors from videos.
2. Convert the descriptors in the so-called sufficient statistics.
3. Convert the sufficient statistics to Fisher vectors or soft-counts.
4. Compute the kernel matrix.
5. Do the training and evaluation.

Descriptor extraction
---------------------

The descriptor extraction stage relies on the dense track code (add
reference). The code is called from Python directly (add function
name) and the extracted descriptors are passed back to Python through
pipeline.

Descriptors to sufficient statistics
------------------------------------

The descriptors are converted to sufficient statistics by the function
[add function name]. This functions requires the previously computed
GMM and PCA. The GMM is stored in the yael format. The PCA is computed
with sklearn and pickled to the disk. The sufficient statistics are
saved to disk to the location specified by [SstatsMap].

Sufficient statistics to Fisher vectors
---------------------------------------

The sufficient statistics are merged and then converted to Fisher
vectors using the function defined in the model class.

Fisher vectors to kernel matrices
---------------------------------

For efficient training the Fisher vectors are dot producted to obtain
kernel matrices for training and testing. Prior to this, we apply the
square rooting transform and the L2 normalization.

Training and evaluation
-----------------------

We use the precomputed kernels for training. Usually the training is
dataset specific.
