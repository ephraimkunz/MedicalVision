#!/usr/bin/env python3
"""
Convert d4data/biomedical-ner-all Hugging Face model to Core ML
for on-device pharmaceutical NER in iOS apps
"""

import coremltools as ct
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

def convert_biomedical_ner_to_coreml():
    """Convert the biomedical NER model to Core ML format"""
    
    # 1. Load the pre-trained model and tokenizer
    model_name = "d4data/biomedical-ner-all"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Get model info
    print(f"Model config: {model.config}")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Label mapping: {model.config.id2label}")
    
    # 2. Set model to evaluation mode
    model.eval()
    
    # 3. Define input specifications for Core ML
    max_sequence_length = 512  # BERT's typical max length
    
    # Create example inputs for tracing
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_sequence_length))
    attention_mask = torch.ones(1, max_sequence_length)
    
    # 4. Create a wrapper model that returns only logits (not dict)
    class NERModelWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Return only logits, not the full output dict
            return outputs.logits
    
    wrapped_model = NERModelWrapper(model)
    wrapped_model.eval()
    
    # Trace the wrapped model
    print("Tracing wrapped PyTorch model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (input_ids, attention_mask),
            strict=False
        )
    
    # 5. Convert to Core ML
    print("Converting to Core ML...")
    
    # Define input types
    input_types = [
        ct.TensorType(
            name="input_ids",
            shape=(1, max_sequence_length),
            dtype=np.int32
        ),
        ct.TensorType(
            name="attention_mask", 
            shape=(1, max_sequence_length),
            dtype=np.int32
        )
    ]
    
    # Convert with proper configuration
    coreml_model = ct.convert(
        traced_model,
        inputs=input_types,
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS18,  # Adjust as needed
        compute_units=ct.ComputeUnit.ALL  # Use Neural Engine when available
    )
    
    # 6. Add metadata
    coreml_model.short_description = "Biomedical Named Entity Recognition"
    coreml_model.author = "d4data/biomedical-ner-all via Hugging Face"
    coreml_model.license = "Check original model license"
    coreml_model.version = "1.0"
    
    # Add input descriptions
    coreml_model.input_description['input_ids'] = "Tokenized input text (BERT tokens)"
    coreml_model.input_description['attention_mask'] = "Attention mask for input tokens"
    coreml_model.output_description['logits'] = "Raw logits for each token classification"
    
    # 7. Save the model
    output_path = "BiomedicalNER.mlpackage"
    coreml_model.save(output_path)
    print(f"Core ML model saved to: {output_path}")
    
    # 8. Save tokenizer and label info for iOS usage
    save_tokenizer_info(tokenizer, model.config)
    
    return coreml_model

def save_tokenizer_info(tokenizer, config):
    """Save tokenizer vocab and label mappings for iOS"""
    import json
    
    # Save vocabulary
    vocab_dict = tokenizer.get_vocab()
    with open("vocab.json", "w") as f:
        json.dump(vocab_dict, f, indent=2)
    
    # Save label mappings
    label_info = {
        "id2label": config.id2label,
        "label2id": config.label2id,
        "num_labels": config.num_labels
    }
    
    with open("labels.json", "w") as f:
        json.dump(label_info, f, indent=2)
    
    # Save special tokens
    special_tokens = {
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "mask_token": tokenizer.mask_token,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "max_length": tokenizer.model_max_length
    }
    
    with open("special_tokens.json", "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    print("Tokenizer info saved:")
    print("- vocab.json (vocabulary mapping)")
    print("- labels.json (entity label mappings)")
    print("- special_tokens.json (special token info)")

    """Test the converted model with sample pharmaceutical text"""
    import json
    import coremltools as ct
    from transformers import AutoTokenizer
    
    # Load the converted model
    model = ct.models.MLModel("BiomedicalNER.mlpackage")
    
    # Load tokenizer and labels for testing
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    
    with open("labels.json", "r") as f:
        label_data = json.load(f)
        id2label = {int(k): v for k, v in label_data["id2label"].items()}
    
    # Sample pharmaceutical text
    sample_text = "Take 25mg of Lisinopril twice daily for hypertension"
    print(f"\nTesting with sample text: '{sample_text}'")
    
    # Tokenize the sample text
    encoded = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    # Prepare inputs for Core ML
    print(encoded)
    input_ids = encoded["input_ids"].astype(np.int32)
    attention_mask = encoded["attention_mask"].astype(np.int32)
    
    # Run prediction
    try:
        prediction = model.predict({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        
        logits = prediction["logits"]
        print(f"Prediction successful! Logits shape: {logits.shape}")
        
        # Get predictions for each token
        predicted_labels = np.argmax(logits[0], axis=-1)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        
        print("\nToken predictions:")
        print("-" * 50)
        
        entities_found = []
        for i, (token, label_id) in enumerate(zip(tokens, predicted_labels)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            label = id2label.get(label_id, "UNKNOWN")
            confidence = np.max(softmax(logits[0][i]))
            
            print(f"{token:15} -> {label:20} (confidence: {confidence:.3f})")
            
            # Collect entities (non-O labels with decent confidence)
            if label != "O" and confidence > 0.5:
                entities_found.append((token, label, confidence))
        
        # Summary
        print(f"\nEntities found: {len(entities_found)}")
        for token, label, conf in entities_found:
            print(f"  '{token}' -> {label} ({conf:.3f})")
            
        if not entities_found:
            print("  No high-confidence entities detected")
            print("  This might indicate:")
            print("  - Model needs fine-tuning on prescription text")
            print("  - Confidence threshold is too high")
            print("  - Text format differs from training data")
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        print("This might indicate:")
        print("- Input shape mismatch")
        print("- Model conversion issues")
        print("- Core ML compatibility problems")

def softmax(x):
    """Apply softmax to get probabilities"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

if __name__ == "__main__":
    # Install required packages first:
    # pip install torch transformers coremltools
    
    try:
        print("Starting biomedical NER model conversion...")
        coreml_model = convert_biomedical_ner_to_coreml()
        print("Conversion completed successfully!")
        
        # Test the conversion
        test_conversion()
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Make sure you have installed: torch, transformers, coremltools")