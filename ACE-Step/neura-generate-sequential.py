import os

# 1. Override the Apple Silicon hard limit to allow SSD swap for memory spikes
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

def initialize_models():
    print("Initializing DiT (Acoustic Renderer)...")
    dit_handler = AceStepHandler()
    
    # 2. Tell the DiT to offload to CPU RAM when it isn't rendering audio
    dit_handler.initialize_service(
        project_root=os.getcwd(),
        config_path="acestep-v15-turbo", 
        device="mps",
        offload_to_cpu=True
    )

    print("Initializing Qwen LM (Composer Agent)...")
    llm_handler = LLMHandler()
    
    # 3. Tell the LM to offload to CPU RAM when it finishes writing the blueprint
    llm_handler.initialize(
        checkpoint_dir="checkpoints",
        lm_model_path="acestep-5Hz-lm-0.6B", 
        backend="pt", 
        device="mps",
        offload_to_cpu=True
    )
    return dit_handler, llm_handler

def main():
    dit, llm = initialize_models()

    # Define the composition blueprint
    params = GenerationParams(
        task_type="text2music",
        caption="ambient generative techno with slow rhythmic pulses and sweeping pads",
        bpm=120,
        duration=20,
        inference_steps=3, 
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
            
        if "masks" in result.extra_outputs:
            mask_tensor = result.extra_outputs["masks"]
            print(f"Extracted Attention Masks Shape: {mask_tensor.shape}")
            
    else:
        print(f"Generation Failed: {result.error}")

if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)
    main()