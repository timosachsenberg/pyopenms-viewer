"""Shared font utilities for rendering modules."""

from PIL import ImageFont

# Common font paths for different systems
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch Linux
]


def get_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font for rendering, with fallbacks.

    Args:
        size: Font size in points

    Returns:
        ImageFont instance (TrueType if available, otherwise default bitmap font)
    """
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
