import traceback  # Add this for detailed error traces

import torch
from transformers import AutoTokenizer

from src.tokens.wrapper import SpecialTokens, TokenizerWrapper


def test_tokenizer_wrapper():
    tokenizer_names = ["bert-base-uncased", "t5-base", "gpt2", "facebook/opt-350m"]

    print("\nTesting TokenizerWrapper...")

    for tokenizer_name in tokenizer_names:
        print(f"\nTesting with {tokenizer_name}")
        try:
            # Initialize tokenizers
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            wrapped = TokenizerWrapper(hf_tokenizer)

            # Test 1: Vocab size matches
            try:
                assert len(wrapped) == len(hf_tokenizer), "Vocab size mismatch"
                print("✓ Vocab size test passed")
            except Exception as e:
                print(f"❌ Vocab size test failed: {str(e)}")
                raise

            # Test 2: Special tokens exist
            try:
                assert wrapped.pad_token_id is not None, "Missing pad token"
                assert wrapped.cls_token_id is not None, "Missing CLS token"
                assert wrapped.sep_token_id is not None, "Missing SEP token"
                assert wrapped.mask_token_id is not None, "Missing mask token"
                print("✓ Special tokens test passed")
            except Exception as e:
                print(f"❌ Special tokens test failed: {str(e)}")
                raise

            # Test 3: Basic encoding/decoding with space normalization
            try:
                test_text = "Hello world!"
                encoded = wrapped.encode(test_text, add_special_tokens=False)
                print(f"Debug - encoded: {encoded}")  # Debug print
                decoded = wrapped.decode(encoded["input_ids"])
                print(f"Debug - decoded: {decoded}")  # Debug print
                norm_decoded = " ".join(decoded.strip().split())
                norm_original = " ".join(test_text.strip().split())
                print(f"Debug - normalized decoded: '{norm_decoded}'")  # Debug print
                print(f"Debug - normalized original: '{norm_original}'")  # Debug print
                assert (
                    norm_decoded == norm_original
                ), "Text mismatch after normalization"
                print("✓ Basic encode/decode test passed")
            except Exception as e:
                print(f"❌ Basic encode/decode test failed: {str(e)}")
                raise

            # Test 4: Batch processing
            try:
                batch_texts = ["Hello world", "Another sentence"]
                encoded_batch = wrapped(batch_texts, padding=True, return_tensors="pt")
                print(
                    f"Debug - batch shape: {encoded_batch['input_ids'].shape}"
                )  # Debug print
                assert isinstance(encoded_batch["input_ids"], torch.Tensor)
                assert encoded_batch["input_ids"].shape[0] == len(batch_texts)
                print("✓ Batch processing test passed")
            except Exception as e:
                print(f"❌ Batch processing test failed: {str(e)}")
                raise

            # Test 5: Max length truncation
            try:
                long_text = " ".join(["word"] * 1000)
                max_length = 10
                encoded_trunc = wrapped.encode(
                    long_text, max_length=max_length, truncation=True
                )
                print(
                    f"Debug - truncated length: {len(encoded_trunc['input_ids'])}"
                )  # Debug print
                assert len(encoded_trunc["input_ids"]) <= max_length
                print("✓ Truncation test passed")
            except Exception as e:
                print(f"❌ Truncation test failed: {str(e)}")
                raise

            # Test 6: Special cases with proper normalization
            try:
                # Empty string
                encoded_empty = wrapped.encode("")
                decoded_empty = wrapped.decode(encoded_empty["input_ids"])
                assert isinstance(decoded_empty, str)

                # Unicode/non-English
                unicode_text = "测试中文"
                encoded_unicode = wrapped.encode(unicode_text)
                decoded_unicode = wrapped.decode(encoded_unicode["input_ids"])
                norm_decoded = "".join(decoded_unicode.strip().split())
                norm_original = "".join(unicode_text.split())
                print(f"Debug - unicode decoded: '{norm_decoded}'")  # Debug print
                print(f"Debug - unicode original: '{norm_original}'")  # Debug print
                assert norm_decoded == norm_original
                print("✓ Special cases test passed")
            except Exception as e:
                print(f"❌ Special cases test failed: {str(e)}")
                raise

        except Exception as e:
            print(f"❌ Tests failed for {tokenizer_name}: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    test_tokenizer_wrapper()
