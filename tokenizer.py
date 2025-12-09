import torch
import jieba
from transformers import AutoTokenizer

class MorphemeAwareTokenizer(AutoTokenizer):
    def __init__(self, pretrained_model_name="hfl/chinese-bert-wwm-ext", **kwargs):
        # Khởi tạo tokenizer HF gốc
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, **kwargs)

    def __len__(self):
        return self.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load tokenizer từ Hugging Face"""
        return cls(pretrained_model_name_or_path, **kwargs)

    # =============================
    # Properties để tương thích với DataCollator
    # =============================
    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def cls_token(self):
        return self.tokenizer.cls_token

    @property
    def sep_token(self):
        return self.tokenizer.sep_token

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors=None, **kwargs):
        """Cho phép DataCollatorForLanguageModeling sử dụng pad()"""
        return self.tokenizer.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs
        )

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """Trả về mask cho special tokens"""
        return self.tokenizer.get_special_tokens_mask(
            token_ids_0=token_ids_0,
            token_ids_1=token_ids_1,
            already_has_special_tokens=already_has_special_tokens
        )

    def convert_tokens_to_ids(self, tokens):
        """Chuyển tokens thành IDs"""
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Chuyển IDs thành tokens"""
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def to_bmes(self, text):
        """
        Tạo danh sách (char, BMES-tag) từ text hoặc list[text].
        Sử dụng jieba để word segmentation.

        Logic:
        - Từ đơn (1 ký tự): gán nhãn 'S'
        - Từ nhiều ký tự: đầu 'B', giữa 'M', cuối 'E'
        """
        if isinstance(text, list):
            return [self.to_bmes(t) for t in text]

        if not isinstance(text, str):
            text = str(text)

        # Sử dụng jieba để segment
        words = list(jieba.cut(text, cut_all=False))

        bmes_list = []
        for word in words:
            n = len(word)
            if n == 1:
                # Từ đơn
                bmes_list.append((word, 'S'))
            else:
                # Từ nhiều ký tự
                bmes_list.append((word[0], 'B'))  # Ký tự đầu
                for char in word[1:-1]:  # Các ký tự giữa
                    bmes_list.append((char, 'M'))
                bmes_list.append((word[-1], 'E'))  # Ký tự cuối

        return bmes_list

    def align_bmes_to_tokens(self, bmes_list, tokens_list):
        """
        Align BMES tags với tokens.
        Xử lý trường hợp token có thể là nhiều ký tự liên tiếp.
        """
        BMES_MAP = {"B": 0, "M": 1, "E": 2, "S": 3}
        aligned_tags = []

        chars = [char for char, tag in bmes_list]
        tags = [tag for char, tag in bmes_list]

        char_idx = 0  # Vị trí hiện tại trong danh sách ký tự

        for token in tokens_list:
            if token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]:
                aligned_tags.append(3)
                continue

            token_len = len(token)

            if char_idx >= len(chars):
                aligned_tags.append(3)
                continue

            if token_len == 1:
                if char_idx < len(tags):
                    aligned_tags.append(BMES_MAP[tags[char_idx]])
                    char_idx += 1
                else:
                    aligned_tags.append(3)
            else:
                token_chars = list(token)
                match = True

                for i, tc in enumerate(token_chars):
                    if char_idx + i >= len(chars) or chars[char_idx + i] != tc:
                        match = False
                        break

                if match:
                    aligned_tags.append(BMES_MAP[tags[char_idx]])
                    char_idx += token_len
                else:
                    aligned_tags.append(3)

        return aligned_tags

    def __call__(self, text, **kwargs):
        BMES_MAP = {"B": 0, "M": 1, "E": 2, "S": 3}

        if isinstance(text, list):
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors=kwargs.get("return_tensors", None),
            )

            bmes_tags_list = []
            for i, t in enumerate(text):
                bmes_list = self.to_bmes(t)
                tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][i].tolist())
                bmes_tags = self.align_bmes_to_tokens(bmes_list, tokens)

                if kwargs.get("return_tensors") == "pt":
                    bmes_tags = torch.tensor(bmes_tags)
                bmes_tags_list.append(bmes_tags)

            if kwargs.get("return_tensors") == "pt":
                max_len = encoded["input_ids"].shape[1]
                padded_bmes = []
                for tags in bmes_tags_list:
                    pad_len = max_len - tags.shape[0]
                    if pad_len > 0:
                        tags = torch.cat([tags, torch.full((pad_len,), 3)])  # Pad với 'S'
                    padded_bmes.append(tags)
                encoded["bmes_tags"] = torch.stack(padded_bmes)
            else:
                encoded["bmes_tags"] = bmes_tags_list

            return encoded

        bmes_list = self.to_bmes(text)
        encoded = self.tokenizer(text, add_special_tokens=True, **kwargs)

        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.squeeze(0).tolist()
        elif isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        bmes_tags = self.align_bmes_to_tokens(bmes_list, tokens)

        if kwargs.get("return_tensors") == "pt":
            bmes_tags = torch.tensor(bmes_tags).unsqueeze(0)

        encoded['bmes_tags'] = bmes_tags
        return encoded

    def save_pretrained(self, save_directory, **kwargs):
        """Save tokenizer"""
        return self.tokenizer.save_pretrained(save_directory, **kwargs)