# YouTube Integration Setup Guide

## YouTube Integration

### Requirements:

1. **yt-dlp**: Python library for downloading YouTube videos
2. **ffmpeg**: Required for audio extraction

### Installation:

```bash
# Install yt-dlp
pip install yt-dlp

# Install ffmpeg (Windows)
# Download from: https://ffmpeg.org/download.html
# Or use chocolatey: choco install ffmpeg

# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Linux)
sudo apt install ffmpeg
```

### Usage:

- Paste any YouTube URL into the app
- The app will download and analyze the audio
- Supports music videos, live performances, etc.

## Installation Commands:

```bash
# Install all dependencies
pip install -r requirement.txt

# Install yt-dlp for YouTube integration
pip install yt-dlp

# For Windows users, also install ffmpeg
# Download from: https://ffmpeg.org/download.html#build-windows
```

## Example URLs:

### YouTube:

- `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- `https://youtu.be/dQw4w9WgXcQ`

## Troubleshooting:

### YouTube Issues:

- Ensure ffmpeg is installed and in PATH
- Some videos may be region-blocked
- Age-restricted content may not work
- Private or deleted videos won't work

### Common Solutions:

1. **"ffmpeg not found"**: Install ffmpeg and add to system PATH
2. **"Video unavailable"**: Try a different video or check if it's region-locked
3. **Download fails**: The video might be private or have restrictions

## Notes:

- Downloaded files are automatically cleaned up after analysis
- Only audio is extracted and analyzed
- Temporary files are deleted immediately after processing
- Respects YouTube's terms of service for personal use
