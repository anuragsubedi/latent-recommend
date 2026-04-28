import os
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

def initialize_models():
    print("Initializing DiT (Acoustic Renderer)...")
    dit_handler = AceStepHandler()
    
    # FIX: Just pass the model name. The handler automatically looks in the 'checkpoints' folder.
    dit_handler.initialize_service(
        project_root=os.getcwd(),
        config_path="acestep-v15-turbo", 
        device="mps" 
    )

    print("Initializing Qwen LM (Composer Agent)...")
    llm_handler = LLMHandler()
    
    # The LLM Handler logic remains exactly the same (it worked perfectly!)
    llm_handler.initialize(
        checkpoint_dir="checkpoints",
        lm_model_path="acestep-5Hz-lm-0.6B", 
        backend="pt", 
        device="mps"
    )
    return dit_handler, llm_handler

def main():
    dit, llm = initialize_models()

    # Define the composition blueprint
    params = GenerationParams(
        task_type="text2music",
        caption="ambient generative techno with slow rhythmic pulses and sweeping pads",
        bpm=110,
        duration=30,
        inference_steps=8, 
        shift=3.0, 
        thinking=True 
    )

    # Configure the hardware execution constraints
    config = GenerationConfig(
        batch_size=1, 
        audio_format="wav" 
    )

    print("Generating audio...")
    result = generate_music(
        dit_handler=dit, 
        llm_handler=llm, 
        params=params, 
        config=config, 
        save_dir="./output"
    )

    if result.success:
        print("\nGeneration Complete!")
        for audio in result.audios:
            print(f"Saved to: {audio['path']}")
            
        # Hook for Mechanistic Interpretability
        if "latents" in result.extra_outputs:
            latent_tensor = result.extra_outputs["latents"]
            print(f"\nExtracted Intermediate Latents Shape: {latent_tensor.shape}")
            # You can dump this tensor to disk here for later layer-level activation analysis
            
        if "masks" in result.extra_outputs:
            mask_tensor = result.extra_outputs["masks"]
            print(f"Extracted Attention Masks Shape: {mask_tensor.shape}")
            
    else:
        print(f"Generation Failed: {result.error}")

if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)
    main()