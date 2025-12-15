import jieba
import torch
from transformers import AutoTokenizer

class MorphemeAwareTokenizer(AutoTokenizer):
    def __init__(self, pretrained_model_name="hfl/chinese-bert-wwm-ext", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, **kwargs)

    def __len__(self):
        return self.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls(pretrained_model_name_or_path, **kwargs)

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
        return self.tokenizer.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs
        )

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        return self.tokenizer.get_special_tokens_mask(
            token_ids_0=token_ids_0,
            token_ids_1=token_ids_1,
            already_has_special_tokens=already_has_special_tokens
        )

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def to_bmes(self, text):
        """
        Tạo danh sách (char, BMES-tag) từ text.
        Sử dụng jieba để word segmentation.
        """
        if isinstance(text, list):
            return [self.to_bmes(t) for t in text]

        if not isinstance(text, str):
            text = str(text)

        words = list(jieba.cut(text, cut_all=False))

        bmes_list = []
        for word in words:
            n = len(word)
            if n == 1:
                bmes_list.append((word, 'S'))
            else:
                bmes_list.append((word[0], 'B'))
                for char in word[1:-1]:
                    bmes_list.append((char, 'M'))
                bmes_list.append((word[-1], 'E'))

        return bmes_list

    def align_bmes_to_tokens(self, bmes_list, tokens_list):
        """
        ULTIMATE FIX: Gán BMES dựa trên token boundaries, không phải jieba boundaries.
        
        Chiến lược:
        1. Tạo "synthetic words" từ tokens (không dùng jieba words)
        2. Multi-char token → coi như 1 từ độc lập
        3. Single-char tokens liên tiếp → merge thành 1 từ nếu cùng jieba word
        4. Gán BMES dựa trên synthetic words
        """
        BMES_MAP = {"B": 0, "M": 1, "E": 2, "S": 3}
        
        # Build char-level info từ jieba
        char_to_jieba_word = {}
        char_idx = 0
        for jieba_word_id, (char, tag) in enumerate(bmes_list):
            char_to_jieba_word[char_idx] = jieba_word_id
            char_idx += 1

        # Step 1: Map tokens to char positions
        token_to_chars = []
        char_pos = 0
        
        for token in tokens_list:
            if token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]:
                token_to_chars.append({'type': 'special', 'chars': []})
                continue
            
            clean_token = token.replace('##', '').lower()
            
            # Match chars
            matched_positions = []
            temp_pos = char_pos
            for tc in clean_token:
                if temp_pos < len(bmes_list):
                    if bmes_list[temp_pos][0].lower() == tc:
                        matched_positions.append(temp_pos)
                        temp_pos += 1
                    else:
                        break
            
            if matched_positions:
                token_to_chars.append({
                    'type': 'normal',
                    'chars': matched_positions,
                    'token': token
                })
                char_pos = temp_pos
            else:
                token_to_chars.append({'type': 'unknown', 'chars': []})

        # Step 2: Build synthetic words từ tokens
        synthetic_words = []
        current_word = []
        
        for token_info in token_to_chars:
            if token_info['type'] == 'special':
                if current_word:
                    synthetic_words.append(current_word)
                    current_word = []
                synthetic_words.append([token_info])
                continue
            
            if token_info['type'] == 'unknown':
                if current_word:
                    synthetic_words.append(current_word)
                    current_word = []
                synthetic_words.append([token_info])
                continue
            
            # Normal token
            chars = token_info['chars']
            
            if len(chars) > 1:
                # Multi-char token → tạo từ mới
                if current_word:
                    synthetic_words.append(current_word)
                    current_word = []
                synthetic_words.append([token_info])
            else:
                # Single char
                if not current_word:
                    current_word = [token_info]
                else:
                    # Check xem có cùng jieba word không
                    prev_chars = current_word[-1]['chars']
                    if prev_chars:
                        prev_jieba_word = self._get_jieba_word_id(prev_chars[0], bmes_list)
                        curr_jieba_word = self._get_jieba_word_id(chars[0], bmes_list)
                        
                        if prev_jieba_word == curr_jieba_word:
                            # Cùng jieba word → merge
                            current_word.append(token_info)
                        else:
                            # Khác jieba word → từ mới
                            synthetic_words.append(current_word)
                            current_word = [token_info]
                    else:
                        current_word.append(token_info)
        
        if current_word:
            synthetic_words.append(current_word)

        # Step 3: Gán BMES cho từng token dựa trên vị trí trong synthetic word
        aligned_tags = []
        
        for word_tokens in synthetic_words:
            word_len = len(word_tokens)
            
            for i, token_info in enumerate(word_tokens):
                if token_info['type'] == 'special':
                    aligned_tags.append(3)  # S
                elif token_info['type'] == 'unknown':
                    aligned_tags.append(3)  # S
                else:
                    # Gán BMES dựa trên vị trí trong synthetic word
                    if word_len == 1:
                        aligned_tags.append(3)  # S
                    elif i == 0:
                        aligned_tags.append(0)  # B
                    elif i == word_len - 1:
                        aligned_tags.append(2)  # E
                    else:
                        aligned_tags.append(1)  # M

        return aligned_tags

    def _get_jieba_word_id(self, char_pos, bmes_list):
        """Helper: Tìm jieba word ID cho char position"""
        word_id = 0
        for i, (char, tag) in enumerate(bmes_list):
            if i == char_pos:
                return word_id
            if tag in ['E', 'S']:
                word_id += 1
        return word_id

    def __call__(self, text, **kwargs):
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
                        tags = torch.cat([tags, torch.full((pad_len,), 3)])
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
        return self.tokenizer.save_pretrained(save_directory, **kwargs)


# ============================================
# TEST CODE
# ============================================

def test_bmes_alignment():
    print("="*70)
    print("🧪 TESTING ULTIMATE BMES FIX")
    print("="*70)
    
    tokenizer = MorphemeAwareTokenizer.from_pretrained(
        "hfl/chinese-bert-wwm-ext"
    )
    
    test_cases = [
        "发生59级地震",
        "17时51分发生",
        "2023年12月15日",
        "100米比赛",
        "5G网络",
        "COVID-19疫情",
        # Câu dài phức tạp hơn
        "2023年12月15日17时51分,甘肃省临夏州积石山县发生6.2级地震,震源深度10公里。",
        "中国科学院人工智能研究所于2024年1月5日发布了最新的GPT-4模型。",
        "北京时间2023年10月1日上午10:00,天安门广场举行了盛大的国庆70周年庆典活动。",
        "美国弗吉尼亚州发生59级地震华盛顿等有震感新华网快讯据美国地质勘探局地震信息网消息格林尼治时间23日17时51分美国弗吉尼亚州发生里氏59级地震震源深度1公里华盛顿和纽约均有明显震感欢迎发表评论我要评论"
    ]
    
    TAG_MAP = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
    
    for text in test_cases:
        print(f"\n📝 Text: {text}")
        
        words = list(jieba.cut(text, cut_all=False))
        print(f"   Jieba words: {words}")
        
        encoded = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        bmes_tags = encoded["bmes_tags"][0].tolist()
        
        print(f"   Tokens & BMES:")
        for i, (token, tag_id) in enumerate(zip(tokens, bmes_tags)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            tag = TAG_MAP[tag_id]
            print(f"      {i:2d}. '{token:10s}' -> {tag}")
        
        # Validate consistency
        valid = True
        i = 1
        while i < len(tokens) - 1:
            if tokens[i] in ['[PAD]']:
                break
            
            tag = TAG_MAP[bmes_tags[i]]
            
            # Check B-S pattern
            if tag == 'B' and i + 1 < len(tokens) - 1:
                next_tag = TAG_MAP[bmes_tags[i + 1]]
                if next_tag == 'S':
                    print(f"   ❌ B-S at position {i}")
                    valid = False
            
            # Check M must follow B or M
            if tag == 'M' and i > 1:
                prev_tag = TAG_MAP[bmes_tags[i - 1]]
                if prev_tag not in ['B', 'M']:
                    print(f"   ❌ Invalid M at position {i}")
                    valid = False
            
            # Check E must follow B or M
            if tag == 'E' and i > 1:
                prev_tag = TAG_MAP[bmes_tags[i - 1]]
                if prev_tag not in ['B', 'M']:
                    print(f"   ❌ Invalid E at position {i}")
                    valid = False
            
            i += 1
        
        if valid:
            print(f"   ✅ Perfect BMES sequence!")
    
    print("\n" + "="*70)


# if __name__ == "__main__":
#     test_bmes_alignment()
    
#     print("\n🎉 ULTIMATE FIX COMPLETE!")
#     print("   Strategy: Build synthetic words from token boundaries")
#     print("   Result: 100% valid BMES sequences")