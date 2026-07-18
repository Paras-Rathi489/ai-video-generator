"""AI video generator: turns a text script into a narrated slideshow video.

Pipeline:
  1. Split script.txt into chunks and use GPT-4 to write one image prompt per chunk.
  2. Generate a stylized image per prompt with gpt-image-1 (with retries + fallbacks).
  3. Align voiceover.mp3 to the script word-by-word using WhisperX.
  4. Assemble a synchronized video (image per chunk, timed to the narration).

Requires the OPENAI_API_KEY environment variable to be set.
"""

import base64
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---------------------------------------------------------

SCRIPT_PATH = "script.txt"
AUDIO_PATH = "voiceover.mp3"
OUTPUT_VIDEO_PATH = "final_video.mp4"
IMAGE_DIR = Path("generated_images")
WHISPER_OUTPUT_DIR = "whisper_output"
CHUNK_SIZE = 28          # words per script chunk / image
MAX_RETRIES = 2          # image generation retries
TEXT_MODEL = "gpt-4"
IMAGE_MODEL = "gpt-image-1"

client = OpenAI()  # reads OPENAI_API_KEY from the environment

# Visual style applied to every generated image
STYLE_JSON = {
    "canvas": {"dimensions": "1024x1024", "background_color": "#000000"},
    "color_palette": {
        "primary": "#FFFFFF",
        "background": "#000000",
        "grayscale": False,
        "transparency": False,
        "color_usage": "none",
    },
    "rendering": {
        "style": "flat",
        "type": "monochrome iconographic",
        "outline_usage": "only when essential for legibility",
        "lighting": "none",
        "shading": "none",
    },
    "subjects": {
        "figures": {
            "form": "highly abstract, often deformed or elongated",
            "features": "dot eyes, straight limbs, no detail",
            "presence": "isolated or symbolic, never naturalistic",
        },
        "objects": {
            "form": "icon-like and blocky",
            "detail_level": "extremely minimal",
            "function": "symbolic representation, not realism",
        },
    },
    "composition": {
        "complexity": "very low",
        "focus": "central or stark left/right split",
        "background_elements": "none",
        "negative_space": "strongly emphasized",
    },
    "stylistic_motifs": [
        "asymmetry", "isolation", "abstraction", "symbol over realism", "emotional void",
    ],
    "effects": {
        "shadows": False,
        "gradients": False,
        "blurs": False,
        "glow": False,
        "textures": False,
        "fades": False,
        "outlines": "hard white only if needed",
    },
    "post_processing_intended": True,
    "purpose": "analog horror symbolic storytelling",
}

STYLE_PREFIX = (
    "Create an image with the following constraints:\n"
    + json.dumps(STYLE_JSON, indent=2)
    + "\nNow illustrate: "
)

# --- Helpers ---------------------------------------------------------------


def create_fallback_image(text: str, size=(1024, 1024), output_path=None) -> Image.Image:
    """Create a simple black image with white text as a placeholder."""
    img = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - (bbox[2] - bbox[0])) // 2
    y = (size[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill="white", font=font)

    if output_path:
        img.save(output_path)
    return img


def split_script_into_chunks(script: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = script.strip().split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def have_all_images(output_dir: Path, total_expected: int) -> bool:
    """Return True if image_01.png ... image_N.png all exist."""
    return all(
        (output_dir / f"image_{i:02d}.png").exists()
        for i in range(1, total_expected + 1)
    )


def missing_image_indices(output_dir: Path, total_expected: int) -> List[int]:
    """Return the 1-based indices of images that are missing."""
    return [
        i for i in range(1, total_expected + 1)
        if not (output_dir / f"image_{i:02d}.png").exists()
    ]


# --- Prompt generation -----------------------------------------------------


def extract_key_context(script_text: str) -> str:
    """Summarize the script's key visual elements for consistent image prompts."""
    print("Analyzing script for key visual context...")
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Analyze this script and extract the key visual elements that should be "
                    "consistent throughout. Focus on: main characters/subjects, their physical "
                    "appearance, setting, and any unique visual traits. Provide a concise "
                    "summary that can be used as context for image generation."
                )},
                {"role": "user", "content": f"Extract key visual context from this script:\n\n{script_text[:3000]}"},
            ],
            temperature=0.3,
        )
        context = response.choices[0].message.content.strip()
        print(f"Key visual context: {context}")
        return context
    except Exception as e:
        print(f"Error extracting context: {e}")
        return "Generic horror/thriller setting with unknown subjects"


def generate_image_prompts_from_script(script_text: str) -> Tuple[List[str], List[str]]:
    """Return (image prompts, script chunks) — one prompt per chunk."""
    print("Generating prompts from script...")
    visual_context = extract_key_context(script_text)
    chunks = split_script_into_chunks(script_text)
    print(f"Total prompts to generate: {len(chunks)}\n")

    system_prompt = f"""You are an assistant that creates visual image prompts for a story.

KEY STORY CONTEXT: {visual_context}

IMPORTANT RULES:
1. Always maintain consistency with the story context above
2. Focus ONLY on visual elements - what can be seen, not dialogue or internal thoughts
3. Keep prompts short and specific - one clear visual scene
4. NEVER include text, words, letters, signs, or writing in the image unless specifically mentioned as a visual element in the story
5. If the chunk mentions specific characters or subjects, use them consistently with their established appearance
6. Describe actions, poses, and scenes, not conversations or explanations
7. Avoid abstract concepts - stick to concrete visual elements

Examples of GOOD prompts:
- "Armless gorilla standing upright in a dark facility"
- "Person running through forest at night"
- "Military vehicles surrounding a building"

Examples of BAD prompts (avoid these):
- "Emergency announcement about evacuation" (too abstract)
- "The concept of genetic enhancement" (not visual)
- "Sign reading 'CongoLand Adventure Park'" (includes text unless specifically mentioned)"""

    prompts = []
    for i, chunk in enumerate(chunks):
        try:
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": (
                        "Create a visual image prompt for this story chunk. "
                        f"Focus on what can be seen:\n\n{chunk}"
                    )},
                ],
                temperature=0.3,
            )
            prompt_text = response.choices[0].message.content.strip().replace('"', "")
            if prompt_text.startswith(("Image: ", "Visual: ")):
                prompt_text = prompt_text.split(": ", 1)[1]

            print(f"[{i + 1}] {prompt_text}")
            prompts.append(prompt_text)
            time.sleep(1.5)  # basic rate limiting
        except Exception as e:
            print(f"Error generating prompt {i + 1}: {e}")
            prompts.append("ERROR")
            time.sleep(5)
    return prompts, chunks


# --- Image generation ------------------------------------------------------


def generate_images_with_fallbacks(
    base_prompts: List[str], output_dir: Path, max_retries: int = MAX_RETRIES
) -> List[str]:
    """Generate one image per prompt, with retries and placeholder fallbacks.

    Images that already exist on disk are skipped, so the script can resume
    a partially completed run. Results are logged to image_log.csv.
    """
    csv_path = output_dir / "image_log.csv"
    image_status = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        logwriter = csv.writer(csvfile)
        logwriter.writerow(["Image Filename", "Prompt", "Status"])

        for idx, base_prompt in enumerate(base_prompts):
            filename = f"image_{idx + 1:02d}.png"
            image_path = output_dir / filename

            if image_path.exists():
                print(f"Skipping {filename} (already exists)")
                logwriter.writerow([filename, base_prompt, "SKIPPED"])
                image_status.append("SKIPPED")
                continue

            if base_prompt == "ERROR":
                print(f"Skipping image {idx + 1} - prompt generation failed")
                create_fallback_image(f"Error: Prompt {idx + 1}", output_path=image_path)
                logwriter.writerow([filename, "FALLBACK - Prompt Error", "FALLBACK"])
                image_status.append("FALLBACK")
                continue

            full_prompt = STYLE_PREFIX + base_prompt
            print(f"\nGenerating image {idx + 1}: {base_prompt}")

            success = False
            for attempt in range(max_retries + 1):
                try:
                    result = client.images.generate(
                        model=IMAGE_MODEL,
                        prompt=full_prompt,
                        size="1024x1024",
                        quality="medium",
                    )
                    image_bytes = base64.b64decode(result.data[0].b64_json)
                    image_path.write_bytes(image_bytes)

                    logwriter.writerow([filename, base_prompt, "SUCCESS"])
                    print(f"Saved: {filename}")
                    image_status.append("SUCCESS")
                    success = True
                    break
                except Exception as e:
                    print(f"Error generating image {idx + 1} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries:
                        print("Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        print(f"Max retries reached for image {idx + 1}, creating fallback")

            if not success:
                fallback_text = f"Image {idx + 1}\n{base_prompt[:50]}..."
                create_fallback_image(fallback_text, output_path=image_path)
                logwriter.writerow([filename, f"FALLBACK - {base_prompt}", "FALLBACK"])
                image_status.append("FALLBACK")
                print(f"Created fallback: {filename}")

            time.sleep(6)  # basic rate limiting between images

    return image_status


def validate_and_fix_images(output_dir: Path, total_expected: int) -> None:
    """Ensure all required images exist; create placeholders for any missing."""
    print(f"\nValidating {total_expected} images...")
    fixed_count = 0
    for i in missing_image_indices(output_dir, total_expected):
        image_path = output_dir / f"image_{i:02d}.png"
        print(f"Creating missing image: {image_path.name}")
        create_fallback_image(f"Missing Image {i}", output_path=image_path)
        fixed_count += 1
    print(f"Fixed {fixed_count} missing images" if fixed_count else "All images present")


# --- Audio alignment -------------------------------------------------------


def run_whisperx_alignment(audio_path: str) -> str:
    """Run WhisperX word-level alignment if not already done; return JSON path."""
    aligned_json_path = f"{WHISPER_OUTPUT_DIR}/{Path(audio_path).stem}.json"
    if Path(aligned_json_path).exists():
        return aligned_json_path

    print("Running WhisperX...")
    try:
        subprocess.run(
            [
                "whisperx", audio_path,
                "--output_dir", WHISPER_OUTPUT_DIR,
                "--model", "large",
                "--language", "en",
                "--compute_type", "int8",
                "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
            ],
            check=True,
        )
        print("WhisperX complete.")
    except subprocess.CalledProcessError as e:
        print(f"WhisperX failed: {e}")
        sys.exit(1)
    return aligned_json_path


def compute_chunk_timings(
    script_chunks: List[str], aligned_json_path: str
) -> List[Tuple[float, float]]:
    """Map each script chunk to a (start, end) time using word-level alignment."""
    with open(aligned_json_path, "r", encoding="utf-8") as f:
        word_segments = json.load(f)["word_segments"]

    chunk_timings = []
    word_idx = 0
    for idx, chunk in enumerate(script_chunks):
        chunk_words = []
        chunk_len = len(chunk.split())

        while len(chunk_words) < chunk_len and word_idx < len(word_segments):
            if word_segments[word_idx]["word"].strip():
                chunk_words.append(word_segments[word_idx])
            word_idx += 1

        if chunk_words:
            start, end = chunk_words[0]["start"], chunk_words[-1]["end"]
            chunk_timings.append((start, end))
            print(f"Chunk {idx + 1}: {start:.2f}s - {end:.2f}s ({end - start:.2f}s)")
        else:
            print(f"No alignment for chunk {idx + 1}: '{chunk[:50]}...'")
    return chunk_timings


# --- Video assembly --------------------------------------------------------


def create_synchronized_video(
    script_chunks: List[str],
    chunk_timings: List[Tuple[float, float]],
    output_dir: Path,
    audio_path: str,
    output_video_path: str,
) -> bool:
    """Assemble the final video: one image per chunk, timed to the narration."""
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration
    clips = []

    print("\nCreating synchronized video...")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Script chunks: {len(script_chunks)}, aligned timings: {len(chunk_timings)}")

    total_aligned_duration = 0
    for i, (start, end) in enumerate(chunk_timings):
        image_path = output_dir / f"image_{i + 1:02d}.png"
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        duration = end - start
        if duration <= 0:
            print(f"Warning: invalid duration {duration}s for segment {i + 1}")
            continue

        clips.append(ImageClip(str(image_path)).set_duration(duration))
        total_aligned_duration += duration
        print(f"Segment {i + 1}: {duration:.2f}s -> {image_path.name}")

    print(f"Total aligned duration: {total_aligned_duration:.2f}s")

    # If narration runs past the last aligned chunk, hold the last image.
    remaining_duration = audio_duration - total_aligned_duration
    if remaining_duration > 0.1:
        print(f"Remaining audio: {remaining_duration:.2f}s")
        extension_image_path = None
        for i in range(len(script_chunks), 0, -1):
            test_path = output_dir / f"image_{i:02d}.png"
            if test_path.exists():
                extension_image_path = test_path
                break

        if extension_image_path:
            print(f"Extending with {extension_image_path.name} for {remaining_duration:.2f}s")
            clips.append(ImageClip(str(extension_image_path)).set_duration(remaining_duration))
        else:
            print("No image available for extension!")

    if not clips:
        print("No valid clips created!")
        return False

    print("Compositing final video...")
    final_video = concatenate_videoclips(clips, method="compose").set_audio(audio)
    final_video.write_videofile(output_video_path, fps=24, verbose=False, logger=None)

    final_duration = sum(clip.duration for clip in clips)
    print(f"Final video duration: {final_duration:.2f}s (audio: {audio_duration:.2f}s)")
    return True


# --- Main workflow ---------------------------------------------------------


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: set the OPENAI_API_KEY environment variable before running.")

    print("Starting video generation workflow...\n")

    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        script_text = f.read()

    base_prompts, script_chunks = generate_image_prompts_from_script(script_text)

    IMAGE_DIR.mkdir(exist_ok=True)
    total_expected = len(script_chunks)

    # Skip generation for images that already exist (resumable runs)
    if have_all_images(IMAGE_DIR, total_expected):
        print("\nFound existing images — skipping generation")
        image_status = ["SKIPPED"] * total_expected
    else:
        missing = missing_image_indices(IMAGE_DIR, total_expected)
        print(f"\nGenerating missing images: {missing}")
        image_status = generate_images_with_fallbacks(base_prompts, IMAGE_DIR)

    validate_and_fix_images(IMAGE_DIR, total_expected)

    # Align narration to script
    print("\nProcessing audio alignment...")
    aligned_json_path = run_whisperx_alignment(AUDIO_PATH)
    chunk_timings = compute_chunk_timings(script_chunks, aligned_json_path)

    # Assemble final video
    success = create_synchronized_video(
        script_chunks, chunk_timings, IMAGE_DIR, AUDIO_PATH, OUTPUT_VIDEO_PATH
    )

    if success:
        print(f"\nSuccess! Final video saved to: {OUTPUT_VIDEO_PATH}")
        print("\nGeneration summary:")
        print(f"  Successful images: {image_status.count('SUCCESS')}")
        print(f"  Fallback images:   {image_status.count('FALLBACK')}")
        print(f"  Total chunks:      {len(script_chunks)}")
        print(f"  Aligned segments:  {len(chunk_timings)}")
    else:
        sys.exit("Video creation failed!")


if __name__ == "__main__":
    main()
