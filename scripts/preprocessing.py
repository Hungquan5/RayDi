import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import random
import logging
from tqdm import tqdm
import pandas as pd
from scipy.signal import resample
import glob

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioAugmenter:
    def __init__(self, sr=16000):
        self.sr = sr
        self.noise_dir = None
        self.cached_noise = None
        self.cache_duration = 30
        
    def set_noise_directory(self, noise_dir):
        """Set the directory containing background noise files"""
        self.noise_dir = noise_dir
        noise_files = glob.glob(str(Path(noise_dir) / "*.wav"))
        logger.info(f"Found {len(noise_files)} background noise files in directory")
        
        # Pre-cache some noise to reduce disk I/O
        if noise_files:
            noise_file = random.choice(noise_files)
            self.cached_noise, _ = sf.read(noise_file)
            # Convert stereo noise to mono if necessary
            if len(self.cached_noise.shape) > 1:
                self.cached_noise = np.mean(self.cached_noise, axis=1)
            if len(self.cached_noise) > self.sr * self.cache_duration:
                self.cached_noise = self.cached_noise[:self.sr * self.cache_duration]
    
    def ensure_mono(self, audio):
        """Convert stereo audio to mono if necessary"""
        if len(audio.shape) > 1:
            return np.mean(audio, axis=1)
        return audio
    
    def match_channels(self, audio, noise):
        """Match noise channels to audio channels"""
        if len(audio.shape) > 1:  # If audio is stereo
            if len(noise.shape) == 1:  # If noise is mono
                noise = np.column_stack((noise, noise))  # Convert noise to stereo
        else:  # If audio is mono
            if len(noise.shape) > 1:  # If noise is stereo
                noise = np.mean(noise, axis=1)  # Convert noise to mono
        return noise
    
    def simple_pitch_shift(self, audio, n_steps):
        """Simplified pitch shift using resampling"""
        factor = 2 ** (n_steps / 12)
        if len(audio.shape) > 1:  # Stereo
            output_length = int(len(audio))
            temp_length = int(output_length / factor)
            
            # Process each channel separately
            stretched_left = resample(audio[:, 0], temp_length)
            stretched_right = resample(audio[:, 1], temp_length)
            
            # Resample back to original length
            final_left = resample(stretched_left, output_length)
            final_right = resample(stretched_right, output_length)
            
            return np.column_stack((final_left, final_right))
        else:  # Mono
            stretched = resample(audio, int(len(audio) / factor))
            return resample(stretched, len(audio))
    
    def simple_time_stretch(self, audio, rate):
        """Simplified time stretch using resampling"""
        if len(audio.shape) > 1:  # Stereo
            new_length = int(len(audio) * rate)
            stretched_left = resample(audio[:, 0], new_length)
            stretched_right = resample(audio[:, 1], new_length)
            return np.column_stack((stretched_left, stretched_right))
        else:  # Mono
            return resample(audio, int(len(audio) * rate))
    
    def add_background_noise(self, audio, noise_level=0.1):
        """Add random background noise using cached noise when possible"""
        if not self.noise_dir or self.cached_noise is None:
            return audio
            
        # Use cached noise if possible
        if len(self.cached_noise) >= len(audio):
            start = random.randint(0, len(self.cached_noise) - len(audio))
            noise = self.cached_noise[start:start + len(audio)]
        else:
            noise = np.tile(self.cached_noise, int(np.ceil(len(audio) / len(self.cached_noise))))
            noise = noise[:len(audio)]
        
        # Match noise channels to audio channels
        noise = self.match_channels(audio, noise)
        
        # Normalize and scale noise
        noise = noise * noise_level
        
        # Combine with original audio
        augmented_audio = audio + noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(augmented_audio))
        if max_val > 0:
            augmented_audio = augmented_audio / max_val
        
        return augmented_audio
    
    def speed_tune(self, audio, speed_factor):
        """Simplified speed tuning"""
        if len(audio.shape) > 1:  # Stereo
            new_length = int(len(audio) / speed_factor)
            speed_left = resample(audio[:, 0], new_length)
            speed_right = resample(audio[:, 1], new_length)
            return np.column_stack((speed_left, speed_right))
        else:  # Mono
            new_length = int(len(audio) / speed_factor)
            return resample(audio, new_length)
    
    def simple_reverb(self, audio, reverberance=50):
        """Simplified reverb effect"""
        delay = int(self.sr * 0.05)
        decay = 0.3 * (reverberance / 100)
        
        if len(audio.shape) > 1:  # Stereo
            delayed = np.zeros_like(audio)
            delayed[delay:] = audio[:-delay] * decay
        else:  # Mono
            delayed = np.zeros_like(audio)
            delayed[delay:] = audio[:-delay] * decay
            
        output = audio + delayed
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val
        return output

    def apply_combined_effects(self, audio, effects_config):
        """Apply multiple effects in sequence, always including noise"""
        augmented = audio.copy()
        
        # Apply each effect in sequence
        for effect, params in effects_config.items():
            if effect == "pitch":
                augmented = self.simple_pitch_shift(augmented, params["n_steps"])
            elif effect == "stretch":
                augmented = self.simple_time_stretch(augmented, params["rate"])
            elif effect == "speed":
                augmented = self.speed_tune(augmented, params["speed"])
            elif effect == "reverb":
                augmented = self.simple_reverb(augmented, params["reverberance"])
        
        # Always apply noise last to ensure it's present
        noise_level = effects_config.get("noise", {}).get("level", random.uniform(0.05, 0.15))
        augmented = self.add_background_noise(augmented, noise_level)
        
        return augmented

def process_audio_file(input_file, output_dir, augmenter):
    """Process a single audio file with combined augmentation"""
    try:
        audio, sr = sf.read(input_file)
        
        # Resample if needed
        if sr != augmenter.sr:
            if len(audio.shape) > 1:  # Stereo
                resampled_left = resample(audio[:, 0], int(len(audio) * augmenter.sr / sr))
                resampled_right = resample(audio[:, 1], int(len(audio) * augmenter.sr / sr))
                audio = np.column_stack((resampled_left, resampled_right))
            else:  # Mono
                audio = resample(audio, int(len(audio) * augmenter.sr / sr))
        
        # Generate random effects configuration
        effects_config = {}
        
        # Randomly select 1-2 additional effects besides noise
        num_effects = random.randint(1, 2)
        possible_effects = ["pitch", "stretch", "speed", "reverb"]
        selected_effects = random.sample(possible_effects, num_effects)
        
        # Configure selected effects
        for effect in selected_effects:
            if effect == "pitch":
                effects_config["pitch"] = {"n_steps": random.uniform(-3, 3)}
            elif effect == "stretch":
                effects_config["stretch"] = {"rate": random.uniform(0.8, 1.2)}
            elif effect == "speed":
                effects_config["speed"] = {"speed": random.uniform(0.9, 1.1)}
            elif effect == "reverb":
                effects_config["reverb"] = {"reverberance": random.randint(30, 70)}
        
        # Always add noise configuration
        effects_config["noise"] = {"level": random.uniform(0.05, 0.15)}
        
        # Apply combined effects
        augmented = augmenter.apply_combined_effects(audio, effects_config)
        
        # Create descriptive suffix for filename
        suffix_parts = []
        for effect, params in effects_config.items():
            if effect == "pitch":
                suffix_parts.append(f"pitch_{abs(int(params['n_steps']))}")
            elif effect == "stretch":
                suffix_parts.append(f"stretch_{int(params['rate']*100)}")
            elif effect == "speed":
                suffix_parts.append(f"speed_{int(params['speed']*100)}")
            elif effect == "reverb":
                suffix_parts.append(f"reverb_{params['reverberance']}")
            elif effect == "noise":
                suffix_parts.append(f"noise_{int(params['level']*100)}")
        
        suffix = "_".join(suffix_parts)
        
        # Create output filename
        input_path = Path(input_file)
        output_file = Path(output_dir) / f"{input_path.stem}_{suffix}{input_path.suffix}"
        
        # Save augmented audio
        sf.write(output_file, augmented, augmenter.sr)
        
        return output_file, effects_config
    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Audio Augmentation Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save augmented audio files")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory containing background noise audio files")
    parser.add_argument("--target_size", type=int, required=True, help="Target number of augmented audio files")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio processing (default: 16000)")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    noise_dir = args.noise_dir
    target_size = args.target_size
    sample_rate = args.sample_rate

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize augmenter
    augmenter = AudioAugmenter(sr=sample_rate)
    augmenter.set_noise_directory(noise_dir)

    # Get list of input files
    input_files = list(Path(input_dir).glob("*.wav"))
    current_size = len(input_files)

    logger.info(f"Current dataset size: {current_size}")
    logger.info(f"Target dataset size: {target_size}")

    # Calculate augmentations needed
    augmentations_per_file = int(np.ceil((target_size - current_size) / current_size))

    # Process files
    processed_files = []
    augmentation_log = []

    logger.info("Starting audio augmentation with combined effects...")
    for input_file in tqdm(input_files, desc="Processing files"):
        for _ in range(augmentations_per_file):
            output_file, effects_config = process_audio_file(
                str(input_file), 
                str(output_dir), 
                augmenter
            )
            if output_file and effects_config:
                processed_files.append(output_file)
                augmentation_log.append({
                    'file_name': output_file.name,
                    'original_file': input_file.stem,
                    'effects_applied': str(effects_config)
                })

    # Count final dataset size
    final_size = len(list(output_dir.glob("*.wav")))
    logger.info(f"Augmentation completed. Final dataset size: {final_size}")

    # Save augmentation log
    pd.DataFrame(augmentation_log).to_csv(output_dir / "augmentation_log.csv", index=False)

if __name__ == "__main__":
    main()
