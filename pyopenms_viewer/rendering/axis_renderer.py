"""Axis rendering for peak maps and ion mobility plots."""

from PIL import Image, ImageDraw, ImageFont

from pyopenms_viewer.annotation.tick_formatter import calculate_nice_ticks, format_tick_label
from pyopenms_viewer.core.state import ViewerState


def get_font(size: int = 12):
    """Get a font, falling back to default if system fonts not available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


class AxisRenderer:
    """Renders axes with ticks and labels on peak map images."""

    def __init__(
        self,
        plot_width: int,
        plot_height: int,
        margin_left: int,
        margin_top: int,
    ):
        """Initialize axis renderer.

        Args:
            plot_width: Width of the plot area
            plot_height: Height of the plot area
            margin_left: Left margin (for y-axis labels)
            margin_top: Top margin
        """
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.margin_left = margin_left
        self.margin_top = margin_top

    def draw(self, canvas: Image.Image, state: ViewerState) -> Image.Image:
        """Draw axes on the canvas.

        Args:
            canvas: PIL Image to draw on (with margins)
            state: ViewerState for bounds and settings

        Returns:
            Canvas with axes drawn
        """
        draw = ImageDraw.Draw(canvas)
        font = get_font(12)
        title_font = get_font(14)

        plot_left = self.margin_left
        plot_right = self.margin_left + self.plot_width
        plot_top = self.margin_top
        plot_bottom = self.margin_top + self.plot_height

        # Draw border
        draw.rectangle(
            [plot_left, plot_top, plot_right, plot_bottom],
            outline=state.axis_color,
            width=1,
        )

        # Get view bounds
        view_rt_min = state.view_rt_min if state.view_rt_min is not None else state.rt_min
        view_rt_max = state.view_rt_max if state.view_rt_max is not None else state.rt_max
        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max

        if state.swap_axes:
            # m/z on x-axis, RT on y-axis
            self._draw_x_axis_mz(draw, font, title_font, state, view_mz_min, view_mz_max, plot_left, plot_bottom)
            self._draw_y_axis_rt(draw, font, title_font, state, view_rt_min, view_rt_max, plot_left, plot_top, canvas)
        else:
            # RT on x-axis, m/z on y-axis
            self._draw_x_axis_rt(draw, font, title_font, state, view_rt_min, view_rt_max, plot_left, plot_bottom)
            self._draw_y_axis_mz(draw, font, title_font, state, view_mz_min, view_mz_max, plot_left, plot_top, canvas)

        return canvas

    def _draw_x_axis_mz(self, draw, font, title_font, state, mz_min, mz_max, plot_left, plot_bottom):
        """Draw x-axis for m/z values."""
        x_ticks = calculate_nice_ticks(mz_min, mz_max, num_ticks=8)
        x_range = mz_max - mz_min

        for tick_val in x_ticks:
            if mz_min <= tick_val <= mz_max:
                x_frac = (tick_val - mz_min) / x_range
                x = plot_left + int(x_frac * self.plot_width)
                draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=state.tick_color, width=1)
                label = format_tick_label(tick_val, x_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                draw.text((x - label_width // 2, plot_bottom + 8), label, fill=state.label_color, font=font)

        x_title = "m/z"
        bbox = draw.textbbox((0, 0), x_title, font=title_font)
        title_width = bbox[2] - bbox[0]
        draw.text(
            (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
            x_title,
            fill=state.label_color,
            font=title_font,
        )

    def _draw_x_axis_rt(self, draw, font, title_font, state, rt_min, rt_max, plot_left, plot_bottom):
        """Draw x-axis for RT values."""
        if state.rt_in_minutes:
            display_rt_min = rt_min / 60.0
            display_rt_max = rt_max / 60.0
            rt_ticks_display = calculate_nice_ticks(display_rt_min, display_rt_max, num_ticks=8)
            rt_ticks = [t * 60.0 for t in rt_ticks_display]
        else:
            rt_ticks = calculate_nice_ticks(rt_min, rt_max, num_ticks=8)
        rt_range = rt_max - rt_min

        for tick_val in rt_ticks:
            if rt_min <= tick_val <= rt_max:
                x_frac = (tick_val - rt_min) / rt_range
                x = plot_left + int(x_frac * self.plot_width)
                draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=state.tick_color, width=1)
                display_val = tick_val / 60.0 if state.rt_in_minutes else tick_val
                display_range = rt_range / 60.0 if state.rt_in_minutes else rt_range
                label = format_tick_label(display_val, display_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                draw.text((x - label_width // 2, plot_bottom + 8), label, fill=state.label_color, font=font)

        x_title = "RT (min)" if state.rt_in_minutes else "RT (s)"
        bbox = draw.textbbox((0, 0), x_title, font=title_font)
        title_width = bbox[2] - bbox[0]
        draw.text(
            (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
            x_title,
            fill=state.label_color,
            font=title_font,
        )

    def _draw_y_axis_rt(self, draw, font, title_font, state, rt_min, rt_max, plot_left, plot_top, canvas):
        """Draw y-axis for RT values."""
        if state.rt_in_minutes:
            display_rt_min = rt_min / 60.0
            display_rt_max = rt_max / 60.0
            y_ticks_display = calculate_nice_ticks(display_rt_min, display_rt_max, num_ticks=8)
            y_ticks = [t * 60.0 for t in y_ticks_display]
        else:
            y_ticks = calculate_nice_ticks(rt_min, rt_max, num_ticks=8)
        y_range = rt_max - rt_min

        for tick_val in y_ticks:
            if rt_min <= tick_val <= rt_max:
                y_frac = 1 - (tick_val - rt_min) / y_range
                y = plot_top + int(y_frac * self.plot_height)
                draw.line([(plot_left - 5, y), (plot_left, y)], fill=state.tick_color, width=1)
                display_val = tick_val / 60.0 if state.rt_in_minutes else tick_val
                display_range = y_range / 60.0 if state.rt_in_minutes else y_range
                label = format_tick_label(display_val, display_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                draw.text(
                    (plot_left - label_width - 10, y - label_height // 2),
                    label,
                    fill=state.label_color,
                    font=font,
                )

        y_title = "RT (min)" if state.rt_in_minutes else "RT (s)"
        self._draw_rotated_y_title(canvas, y_title, title_font, state, plot_top)

    def _draw_y_axis_mz(self, draw, font, title_font, state, mz_min, mz_max, plot_left, plot_top, canvas):
        """Draw y-axis for m/z values."""
        mz_ticks = calculate_nice_ticks(mz_min, mz_max, num_ticks=8)
        mz_range = mz_max - mz_min

        for tick_val in mz_ticks:
            if mz_min <= tick_val <= mz_max:
                y_frac = 1 - (tick_val - mz_min) / mz_range
                y = plot_top + int(y_frac * self.plot_height)
                draw.line([(plot_left - 5, y), (plot_left, y)], fill=state.tick_color, width=1)
                label = format_tick_label(tick_val, mz_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                draw.text(
                    (plot_left - label_width - 10, y - label_height // 2),
                    label,
                    fill=state.label_color,
                    font=font,
                )

        self._draw_rotated_y_title(canvas, "m/z", title_font, state, plot_top)

    def _draw_rotated_y_title(self, canvas, title, font, state, plot_top):
        """Draw rotated y-axis title."""
        txt_img = Image.new("RGBA", (100, 30), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), title, fill=state.label_color, font=font)
        txt_img = txt_img.rotate(90, expand=True)

        y_title_x = 5
        y_title_y = plot_top + self.plot_height // 2 - txt_img.height // 2
        canvas.paste(txt_img, (y_title_x, y_title_y), txt_img)


class IMAxisRenderer:
    """Renders axes for ion mobility plots."""

    def __init__(
        self,
        plot_width: int,
        plot_height: int,
        margin_left: int,
        margin_top: int,
    ):
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.margin_left = margin_left
        self.margin_top = margin_top

    def draw(self, canvas: Image.Image, state: ViewerState) -> Image.Image:
        """Draw IM axes on the canvas.

        Args:
            canvas: PIL Image to draw on
            state: ViewerState for bounds and settings

        Returns:
            Canvas with axes drawn
        """
        draw = ImageDraw.Draw(canvas)
        font = get_font(12)
        title_font = get_font(14)

        plot_left = self.margin_left
        plot_right = self.margin_left + self.plot_width
        plot_top = self.margin_top
        plot_bottom = self.margin_top + self.plot_height

        # Draw border
        draw.rectangle(
            [plot_left, plot_top, plot_right, plot_bottom],
            outline=state.axis_color,
            width=1,
        )

        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max
        view_im_min = state.view_im_min if state.view_im_min is not None else state.im_min
        view_im_max = state.view_im_max if state.view_im_max is not None else state.im_max

        # X-axis: m/z
        mz_ticks = calculate_nice_ticks(view_mz_min, view_mz_max, num_ticks=8)
        mz_range = view_mz_max - view_mz_min

        for tick_val in mz_ticks:
            if view_mz_min <= tick_val <= view_mz_max:
                x_frac = (tick_val - view_mz_min) / mz_range
                x = plot_left + int(x_frac * self.plot_width)
                draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=state.tick_color, width=1)
                label = format_tick_label(tick_val, mz_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                draw.text((x - label_width // 2, plot_bottom + 8), label, fill=state.label_color, font=font)

        # X-axis title
        x_title = "m/z"
        bbox = draw.textbbox((0, 0), x_title, font=title_font)
        title_width = bbox[2] - bbox[0]
        draw.text(
            (plot_left + self.plot_width // 2 - title_width // 2, plot_bottom + 28),
            x_title,
            fill=state.label_color,
            font=title_font,
        )

        # Y-axis: IM
        im_ticks = calculate_nice_ticks(view_im_min, view_im_max, num_ticks=8)
        im_range = view_im_max - view_im_min

        for tick_val in im_ticks:
            if view_im_min <= tick_val <= view_im_max:
                y_frac = 1 - (tick_val - view_im_min) / im_range
                y = plot_top + int(y_frac * self.plot_height)
                draw.line([(plot_left - 5, y), (plot_left, y)], fill=state.tick_color, width=1)
                label = format_tick_label(tick_val, im_range)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_width = bbox[2] - bbox[0]
                label_height = bbox[3] - bbox[1]
                draw.text(
                    (plot_left - label_width - 10, y - label_height // 2),
                    label,
                    fill=state.label_color,
                    font=font,
                )

        # Y-axis title
        y_title = f"IM ({state.im_unit})" if state.im_unit else "IM"
        txt_img = Image.new("RGBA", (100, 30), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((0, 0), y_title, fill=state.label_color, font=title_font)
        txt_img = txt_img.rotate(90, expand=True)

        y_title_x = 5
        y_title_y = plot_top + self.plot_height // 2 - txt_img.height // 2
        canvas.paste(txt_img, (y_title_x, y_title_y), txt_img)

        return canvas
