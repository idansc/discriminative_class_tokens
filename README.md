# Discriminative Class Tokens for Text-to-Image Diffusion Models (ICCV'2023)

This repository contains the code related to our paper *Discriminative Class Tokens for Text-to-Image Diffusion models*.

> Idan Schwartz\*<sup>1</sup>, Vésteinn Snæbjarnarson\*<sup>2</sup>, Hila Chefer<sup>1</sup>, Serge Belongie<sup>2</sup>, Lior Wolf<sup>1</sup>, Sagie Benaim<sup>2</sup>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>University of Copenhagen, <sup>3</sup>ETH Zürich
> \* Denotes equal contribution  
>
> Recent advances in text-to-image diffusion models have enabled the generation of diverse and high-quality images. However, generated images often fall short of depicting subtle details and are susceptible to errors due to ambiguity in the input text. One way of alleviating these issues is to train diffusion models on class-labeled datasets. This comes with a downside, doing so limits their expressive power: (i) supervised datasets are generally small compared to large-scale scraped text-image datasets on which text-to-image models are trained, and so the quality and diversity of generated images are severely affected, or (ii) the input is a hard-coded label, as opposed to free-form text, which limits the control over the generated images. In this work, we propose a non-invasive fine-tuning technique that capitalizes on the expressive potential of free-form text while achieving high accuracy through discriminative signals from a pretrained classifier, which guides the generation. This is done by iteratively modifying the embedding of a single input token of a text-to-image diffusion model, using the classifier, by steering generated images toward a given target class. Our method is fast compared to prior fine-tuning methods and does not require a collection of in-class images or retraining of a noise-tolerant classifier. We evaluate our method extensively, showing that the generated images are: (i) more accurate and of higher quality than standard diffusion models, (ii) can be used to augment training data in a low-resource setting, and (iii) reveal information about the data used to train the guiding classifier.
> 
<a href="https://arxiv.org/abs/2303.17155"><img src="https://img.shields.io/badge/arXiv-2303.17155-b31b1b.svg" height=30.5></a> <a href="https://vesteinn.github.io/disco/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=30.5></a> <a href="https://colab.research.google.com/drive/1xl3_BjSPTT8D9GTsO6wZxziAuKkC9fKi?usp=sharing"><img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" height=30.5></a> 


<p align="center">
<img src="https://github.com/idansc/discriminative_class_tokens/blob/main/docs/teaser.jpg" width="800px"/>  
<br>
We propose a technique that introduces a token ($S_c$) corresponding to an external classifier label class $c$. This improves text-to-image alignment when there is lexical ambiguity and enhances the depiction of intricate details.
</p>


## TODO:
- [x] Release code and support for ImageNet
- [x] Release support for iNaturalist and CUB200
- [x] Add google colab
- []  Add hf-spaces 

#### installations:

`conda env create -f requirements.yml`

`conda activate discriminative-token`

Run this command to log in with your HF Hub token if you haven't before:

`huggingface-cli login`

## Run and Evaluate:
<p align="center">
<img src="https://github.com/idansc/discriminative_class_tokens/blob/main/docs/method.jpg" width="450px"/>  
<br>
An overview of our method for optimizing a new discriminative token representation ($v_c$) using a pre-trained classifier. For the prompt `A photo of a $S_c$ tiger cat,' we expect the output generated with the class $c$ to be `tiger cat.' The classifier, however, indicates that the class of the generated image is a `tiger'. We generate images iteratively and optimize the token representation using cross-entropy. Once $v_c$ has been trained, more images of the target class can be generated by including it in the context of the input text.
</p>


To train and evaluate use:
`python run.py --class_index 283 --train True  --evaluate True`

#### Hyperparameters:
The hyperparameters can be changed in the `config.py` script. Note that the paper results are based on stable-diffusion version 1.4.

#### Outputs
The script will create folders and store tokens representation in `pipeline_token` and the images in `img.` 


## Citation

If you make use of our work, please cite our paper:

```
@article{schwartz2023discriminative,
  title={Discriminative Class Tokens for Text-to-Image Diffusion Models},
  author={Schwartz, Idan and Sn{\ae}bjarnarson, V{\'e}steinn and Chefer, Hila and Belongie, Serge and Wolf, Lior and Benaim, Sagie},
  journal={arXiv preprint arXiv:2303.17155},
  year={2023}
}
```
