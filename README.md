# FLD
Normalizing Flow based Evaluation Metric for Generative Models

We propose two new evaluation metrics to assess realness of generated images based on normalizing flows: a simpler and efficient flow-based likelihood distance (FLD) and a more exact dual-flow based likelihood distance (D-FLD). Because normalizing flows can be used to compute the exact likelihood, the proposed metrics assess how closely generated images align with the distribution of real images from a given domain. This property gives the proposed metrics a few advantages over the widely used Fréchet inception distance (FID) and other recent metrics. Firstly, the proposed metrics need only a few hundred images to stabilize (converge in mean), as opposed to tens of thousands needed for FID, and at least a few thousand for the other metrics. This allows confident evaluation of even small sets of generated images, such as validation batches inside training loops. Secondly, the network used to compute the proposed metric has over an order of magnitude fewer parameters compared to Inception-V3 used to compute FID, making it computationally more efficient. For assessing the realness of generated images in new domains (e.g., x-ray images), ideally these networks should be retrained on real images to model their distinct distributions. Thus, our smaller network will be even more advantageous for new domains. Extensive experiments show that the proposed metrics have the desired monotonic relationships with the extent of image degradation of various kinds.

![Screenshot 2024-10-04 160854](https://github.com/user-attachments/assets/8d967143-7ce7-46b2-8502-34d5bc5982f6)

![image](https://github.com/user-attachments/assets/bc10bfad-f126-473b-9924-b8c588212b54)


## Citation
If you found this code helpful, please consider citing: 
```
@misc{jeevan2024normalizingflowbasedmetric,
      title={Normalizing Flow Based Metric for Image Generation}, 
      author={Pranav Jeevan and Neeraj Nixon and Amit Sethi},
      year={2024},
      eprint={2410.02004},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.02004}, 
}

```
