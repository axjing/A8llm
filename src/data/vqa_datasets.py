
import logging
from typing import Any
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.processors import get_image_string
class DatasetBase(Dataset):
    def __init__(self,dataset, tokenizer, image_processor, mp_image_token_length, relevance_min_rating=1, image_correspondence_min_rating=1, visual_dependency_min_rating=1, formatting_min_rating=1) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        self.prefix_len = self._get_prefix_len()
    
    def __len__(self):
        return len(self.dataset)
    
    def _get_prefix_len(self):
        random_string_5_letters='xzyvd'
        random_string_chat_templated=self.tokenizer.apply_chat_template(
            [
                {
                    'role':'assistant',
                    'content':random_string_5_letters,
                }
            ],
            tokenize=False,
            add_special_tokens=False
        )
        random_string_location=random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))
    def _get_messages(self,item,splitted_image_counts):
        messages=[]
        for index,text in enumerate(item['texts']):
            try:
                if item.get('relevance_ratings') is not None and item['relevance_ratings'][index] is not None and item['relevance_ratings'][index] < self.relevance_min_rating:
                    continue
                if item.get('image_correspondence_ratings') is not None and item['image_correspondence_ratings'][index] is not None and item['image_correspondence_ratings'][index] < self.image_correspondence_min_rating:
                    continue
                if item.get('visual_dependency_ratings') is not None and item['visual_dependency_ratings'][index] is not None and item['visual_dependency_ratings'][index] < self.visual_dependency_min_rating:
                    continue
                if item.get('formatting_ratings') is not None and item['formatting_ratings'][index] is not None and item['formatting_ratings'][index] < self.formatting_min_rating:
                    continue
            except Exception as e:
                logging.warning(f"Error processing item: {item}, index: {index}: {e}")
            
            messages.append({'role':'user','content':text['user']})
            messages.append({'role':'assistant','content':text['assistant']})
        
        if len(messages)==0:
            return messages
        
        # Safety check to ensure no image tokens are persent in the text before adding them.
        for msg in messages:
            if self.tokenizer.image_token in msg['content']:
                logging.warning(f"Found and removed an image token in the {msg['role']} text before adding the image string.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")
            
        if len(splitted_image_counts)>0:
            image_string=get_image_string(self.tokenizer,splitted_image_counts,self.mp_image_token_length)
            messages[0]['content']=image_string+messages[0]['content']
        
        return messages
    def _process_images(self, images):
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image, splitted_image_count = self.image_processor(image)
                if not hasattr(self.tokenizer, "global_image_token") and splitted_image_count[0]*splitted_image_count[1] == len(processed_image) - 1:
                    # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                    processed_image = processed_image[1:]
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise ValueError(f"Error processing image: {image}")
        return processed_images, splitted_image_counts
    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end   = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return torch.tensor(conv_ids["input_ids"]), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])
            
class VQADataset(DatasetBase):  # Visual Question Answering Dataset
    def iter_for_worker(self):  # with iterable datasets, each worker gets different shards
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item):
        # Handle images (should be a list)
        if item['images'] is None:
            images_data = []
        else:
            images_data = item['images']
            if not isinstance(images_data, list):
                images_data = [images_data]

        processed_images = []
        splitted_image_counts = []
        if images_data: # Only process if there are images
            processed_images, splitted_image_counts = self._process_images(images_data)

        messages = self._get_messages(item, splitted_image_counts)

        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1) # Shift labels for causal LM
        labels[-1] = -100 # Last token has no target
        
        return labels
    
class CollatorBase(object):
    def __init__(self,tokenizer) -> None:
        self.tokenizer=tokenizer
        
        self.data_field={"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        
    def _pad_batch(self,batch:dict,max_length:int):
        batch['input_ids']=[torch.nn.functional.pad(ids,(max_length-len(ids),0),value=self.tokenizer.pad_token_id) for ids in batch['input_ids']]
        batch['labels']=[torch.nn.functional.pad(labels,(max_length-len(labels),0),value=self.tokenizer.pad_token_id) for labels in batch['labels']]
        batch['attention_mask']=[torch.nn.functional.pad(attention_mask,(max_length-len(attention_mask),0),value=0) for attention_mask in batch['attention_mask']]
    
    def prepare_batch(self,batch,max_lenght=None):
        # 1. Hadndle empty
        if not batch:
            return self.data_field
        
        # 2. Drop None rows
        batch=[s for s in batch if s is not None]
        if not batch:
            return self.data_field
        
        # 3. batch is a list of dicts, each containing 'input_ids', 'attention_mask', 'labels', 'images'
        # let's convert it to a dict of lists of tensors
        batch={k:[item[k] for item in batch] for k in batch[0]}
        
        if max_lenght is not None:
            batch=self._discard_samples_that_are_too_long(batch,max_lenght)
            
        if len(batch['input_ids'])==0:
            return batch
        
        # 4. Pad samples to max_length
        if max_lenght is not None:
            max_len=max_lenght
        else:
            max_len=max(map(len,batch['input_ids']))
        
        self._pad_batch(batch,max_len)
        
        return {
            'input_ids':torch.stack(batch['input_ids']),
            'attention_mask':torch.stack(batch['attention_mask']),
            'images':batch['images'],
            'labels':torch.stack(batch['labels']),
        }
            
    
    def _discard_samples_that_are_too_long(self,batch,max_length:int):
        filtered=[
            (ids,label,attn_mask,image)
            for ids,label,attn_mask,image in zip(batch['input_ids'],batch['labels'],batch['attention_mask'],batch['images']) if len(ids) <=max_length
        ]
        
        if not filtered:
            return self.data_field
        
        batch_token_ids,batch_labels,batch_attention_mask,batch_images=zip(*filtered)
        
        return{'input_ids':list(batch_token_ids),'labels':list(batch_labels),'attention_mask':list(batch_attention_mask),'images':list(batch_images)}
        
class VQACollator(CollatorBase) :
    def __init__(self, tokenizer,max_length) -> None:
        self.max_length=max_length
        super().__init__(tokenizer)
        
    def _pad_batch(self, batch: dict, max_length: int):
        # 重新改写，将标签的填充值设为 -100，这样损失函数会自动忽略该值。
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]
    
    def __call__(self, batch:dict) -> Any:
        batch=self.prepare_batch(batch,max_lenght=self.max_length)
        return batch