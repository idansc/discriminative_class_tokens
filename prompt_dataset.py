from torch.utils.data import Dataset


imagenet_templates_small = [
    "A photo of {}",
    "a high resolution realistic image of {}",
    "A close-up photo of a {}",
    "A rendition of a {}",
    "a photo of a nice {}",
    "A painting of a {}",
    "a good photo of a {}",
    "the photo of a {}",
    "a bright photo of the {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


class PromptDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the promots for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        placeholder_token,
        prompt_suffix,
        tokenizer,
        epoch_size,
        number_of_prompts,
    ):
        self.prompt_suffix = prompt_suffix
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.epoch_size = epoch_size
        self.number_of_prompts = number_of_prompts

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        example = {}
        text = imagenet_templates_small[index % self.number_of_prompts]
        text = text.format(self.placeholder_token)
        text += f" {self.prompt_suffix}"
        example["instance_prompt"] = text
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return example
