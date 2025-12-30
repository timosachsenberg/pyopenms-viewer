"""Overlay rendering for features, IDs, and markers on peak maps."""

from PIL import Image, ImageDraw, ImageFont

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


class OverlayRenderer:
    """Renders overlays (features, IDs, markers) on peak map images."""

    def __init__(
        self,
        plot_width: int,
        plot_height: int,
        margin_left: int = 0,
        margin_top: int = 0,
    ):
        """Initialize overlay renderer.

        Args:
            plot_width: Width of the plot area
            plot_height: Height of the plot area
            margin_left: Left margin (for coordinate calculations)
            margin_top: Top margin (for coordinate calculations)
        """
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.margin_left = margin_left
        self.margin_top = margin_top

    def data_to_plot_pixel(self, state: ViewerState, rt: float, mz: float) -> tuple[int, int]:
        """Convert RT/m/z to plot pixel coordinates.

        Args:
            state: ViewerState with current view bounds
            rt: Retention time value
            mz: m/z value

        Returns:
            Tuple of (x, y) pixel coordinates within the plot area
        """
        view_rt_min = state.view_rt_min if state.view_rt_min is not None else state.rt_min
        view_rt_max = state.view_rt_max if state.view_rt_max is not None else state.rt_max
        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max

        rt_range = view_rt_max - view_rt_min
        mz_range = view_mz_max - view_mz_min

        if rt_range == 0 or mz_range == 0:
            return (0, 0)

        if state.swap_axes:
            # m/z on x-axis, RT on y-axis (inverted)
            x = int((mz - view_mz_min) / mz_range * self.plot_width)
            y = int((1 - (rt - view_rt_min) / rt_range) * self.plot_height)
        else:
            # RT on x-axis, m/z on y-axis (inverted)
            x = int((rt - view_rt_min) / rt_range * self.plot_width)
            y = int((1 - (mz - view_mz_min) / mz_range) * self.plot_height)

        return (x, y)

    def feature_intersects_view(
        self, state: ViewerState, rt_min: float, rt_max: float, mz_min: float, mz_max: float
    ) -> bool:
        """Check if a feature bounding box intersects the current view."""
        view_rt_min = state.view_rt_min if state.view_rt_min is not None else state.rt_min
        view_rt_max = state.view_rt_max if state.view_rt_max is not None else state.rt_max
        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max

        return rt_max >= view_rt_min and rt_min <= view_rt_max and mz_max >= view_mz_min and mz_min <= view_mz_max

    def draw_features(self, img: Image.Image, state: ViewerState) -> Image.Image:
        """Draw feature overlays (centroids, bounding boxes, convex hulls).

        Args:
            img: Plot image to draw on (without margins)
            state: ViewerState with feature data

        Returns:
            Image with features drawn
        """
        if state.feature_map is None or state.feature_map.size() == 0:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        max_features = 10000
        features_drawn = 0

        for idx, feature in enumerate(state.feature_map):
            if features_drawn >= max_features:
                break

            is_selected = idx == state.selected_feature_idx
            is_hovered = idx == state.hover_feature_idx
            rt = feature.getRT()
            mz = feature.getMZ()

            # Get feature bounds from convex hulls
            hulls = feature.getConvexHulls()
            if hulls:
                all_points = []
                for hull in hulls:
                    points = hull.getHullPoints()
                    all_points.extend([(p[0], p[1]) for p in points])

                if all_points:
                    rt_coords = [p[0] for p in all_points]
                    mz_coords = [p[1] for p in all_points]
                    feat_rt_min, feat_rt_max = min(rt_coords), max(rt_coords)
                    feat_mz_min, feat_mz_max = min(mz_coords), max(mz_coords)
                else:
                    feat_rt_min, feat_rt_max = rt - 1, rt + 1
                    feat_mz_min, feat_mz_max = mz - 0.5, mz + 0.5
            else:
                feat_rt_min, feat_rt_max = rt - 1, rt + 1
                feat_mz_min, feat_mz_max = mz - 0.5, mz + 0.5

            if not self.feature_intersects_view(state, feat_rt_min, feat_rt_max, feat_mz_min, feat_mz_max):
                continue

            features_drawn += 1

            # Colors - priority: selected > hovered > default
            if is_selected:
                hull_color = state.selected_color
                bbox_color = state.selected_color
                centroid_color = state.selected_color
                line_width = 3
            elif is_hovered:
                hull_color = state.hover_color
                bbox_color = state.hover_color
                centroid_color = state.hover_color
                line_width = 2
            else:
                hull_color = state.hull_color
                bbox_color = state.bbox_color
                centroid_color = state.centroid_color
                line_width = 1

            # Draw convex hulls
            if state.show_convex_hulls and hulls:
                for hull in hulls:
                    points = hull.getHullPoints()
                    if len(points) >= 3:
                        pixel_points = [self.data_to_plot_pixel(state, p[0], p[1]) for p in points]
                        pixel_points.append(pixel_points[0])
                        fill_alpha = 100 if is_selected else 50
                        draw.polygon(pixel_points, outline=hull_color, fill=(*hull_color[:3], fill_alpha))

            # Draw bounding boxes
            if state.show_bounding_boxes:
                top_left = self.data_to_plot_pixel(state, feat_rt_min, feat_mz_max)
                bottom_right = self.data_to_plot_pixel(state, feat_rt_max, feat_mz_min)
                x1, y1 = top_left
                x2, y2 = bottom_right
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=line_width)

            # Draw centroids
            if state.show_centroids:
                cx, cy = self.data_to_plot_pixel(state, rt, mz)

                # Draw hover glow ring first (behind the centroid)
                if is_hovered and not is_selected:
                    glow_r = 8
                    # Outer glow ring
                    draw.ellipse(
                        [cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
                        outline=(*state.hover_color[:3], 150),
                        width=2,
                    )

                # Centroid size: selected > hovered > default
                if is_selected:
                    r = 6
                    outline_width = 2
                elif is_hovered:
                    r = 5
                    outline_width = 2
                else:
                    r = 3
                    outline_width = 1

                draw.ellipse(
                    [cx - r, cy - r, cx + r, cy + r],
                    fill=centroid_color,
                    outline=(255, 255, 255, 255),
                    width=outline_width,
                )

        img = Image.alpha_composite(img, overlay)
        return img

    def draw_ids(self, img: Image.Image, state: ViewerState) -> Image.Image:
        """Draw peptide ID markers.

        Args:
            img: Plot image to draw on
            state: ViewerState with ID data

        Returns:
            Image with ID markers drawn
        """
        if not state.peptide_ids or not state.show_ids:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = None
        if state.show_id_sequences:
            font = get_font(8)

        max_ids = 5000
        ids_drawn = 0

        for idx, pep_id in enumerate(state.peptide_ids):
            if ids_drawn >= max_ids:
                break

            rt = pep_id.getRT()
            mz = pep_id.getMZ()

            # Check if in view
            view_rt_min = state.view_rt_min if state.view_rt_min is not None else state.rt_min
            view_rt_max = state.view_rt_max if state.view_rt_max is not None else state.rt_max
            view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
            view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max

            if not (view_rt_min <= rt <= view_rt_max and view_mz_min <= mz <= view_mz_max):
                continue

            ids_drawn += 1

            is_selected = idx == state.selected_id_idx
            x, y = self.data_to_plot_pixel(state, rt, mz)

            color = state.id_selected_color if is_selected else state.id_color
            r = 6 if is_selected else 4

            # Draw diamond shape
            draw.polygon(
                [
                    (x, y - r),
                    (x + r, y),
                    (x, y + r),
                    (x - r, y),
                ],
                fill=color,
                outline=(255, 255, 255, 255),
            )

            # Draw sequence label if enabled
            if state.show_id_sequences and font:
                hits = pep_id.getHits()
                if hits:
                    seq = hits[0].getSequence().toString()
                    short_seq = seq[:10] + "..." if len(seq) > 10 else seq
                    draw.text((x + r + 2, y - 5), short_seq, fill=color, font=font)

        img = Image.alpha_composite(img, overlay)
        return img

    def draw_spectrum_marker(self, img: Image.Image, state: ViewerState) -> Image.Image:
        """Draw crosshair at selected spectrum position.

        Args:
            img: Plot image to draw on
            state: ViewerState with spectrum selection

        Returns:
            Image with marker drawn
        """
        if not state.show_spectrum_marker:
            return img
        if state.selected_spectrum_idx is None or state.exp is None:
            return img

        spec = state.exp[state.selected_spectrum_idx]
        rt = spec.getRT()
        ms_level = spec.getMSLevel()

        # Get precursor m/z for MS2
        precursor_mz = None
        if ms_level == 2:
            precursors = spec.getPrecursors()
            if precursors:
                precursor_mz = precursors[0].getMZ()

        # Check if in view
        view_rt_min = state.view_rt_min if state.view_rt_min is not None else state.rt_min
        view_rt_max = state.view_rt_max if state.view_rt_max is not None else state.rt_max
        view_mz_min = state.view_mz_min if state.view_mz_min is not None else state.mz_min
        view_mz_max = state.view_mz_max if state.view_mz_max is not None else state.mz_max

        rt_in_view = view_rt_min <= rt <= view_rt_max
        mz_in_view = precursor_mz is not None and view_mz_min <= precursor_mz <= view_mz_max

        if not rt_in_view and not mz_in_view:
            return img

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Colors
        line_color = (0, 212, 255, 200) if ms_level == 1 else (255, 107, 107, 200)
        crosshair_color = (255, 200, 50, 180)

        font = get_font(10)

        x, y = self.data_to_plot_pixel(state, rt, view_mz_min if not state.swap_axes else view_mz_max)

        # Draw RT line
        if rt_in_view:
            if state.swap_axes:
                # Horizontal line
                draw.line([(0, y - 1), (self.plot_width, y - 1)], fill=line_color, width=1)
                draw.line([(0, y + 1), (self.plot_width, y + 1)], fill=line_color, width=1)
                label = f"MS{ms_level} #{state.selected_spectrum_idx}"
                draw.text((4, y + 4), label, fill=line_color, font=font)
            else:
                # Vertical line
                draw.line([(x - 1, 0), (x - 1, self.plot_height)], fill=line_color, width=1)
                draw.line([(x + 1, 0), (x + 1, self.plot_height)], fill=line_color, width=1)
                label = f"MS{ms_level} #{state.selected_spectrum_idx}"
                draw.text((x + 4, 4), label, fill=line_color, font=font)

        # Draw precursor m/z line for MS2
        if precursor_mz is not None and mz_in_view:
            prec_x, prec_y = self.data_to_plot_pixel(state, rt, precursor_mz)

            if state.swap_axes:
                draw.line([(prec_x, 0), (prec_x, self.plot_height)], fill=line_color, width=2)
                mz_label = f"Prec: {precursor_mz:.4f}"
                draw.text((prec_x + 4, 4), mz_label, fill=line_color, font=font)
            else:
                draw.line([(0, prec_y), (self.plot_width, prec_y)], fill=line_color, width=2)
                mz_label = f"Prec: {precursor_mz:.4f}"
                bbox = draw.textbbox((0, 0), mz_label, font=font)
                label_width = bbox[2] - bbox[0]
                draw.text((self.plot_width - label_width - 4, prec_y - 14), mz_label, fill=line_color, font=font)

            # Draw crosshair intersection
            if rt_in_view:
                r = 6
                draw.ellipse([(prec_x - r, prec_y - r), (prec_x + r, prec_y + r)], outline=crosshair_color, width=2)
                draw.line([(prec_x - r - 2, prec_y), (prec_x + r + 2, prec_y)], fill=crosshair_color, width=1)
                draw.line([(prec_x, prec_y - r - 2), (prec_x, prec_y + r + 2)], fill=crosshair_color, width=1)

        img = Image.alpha_composite(img, overlay)
        return img

    def draw_all(self, img: Image.Image, state: ViewerState) -> Image.Image:
        """Draw all overlays (features, IDs, spectrum marker).

        Args:
            img: Plot image to draw on
            state: ViewerState with all overlay data

        Returns:
            Image with all overlays drawn
        """
        if state.feature_map is not None:
            img = self.draw_features(img, state)
        if state.peptide_ids:
            img = self.draw_ids(img, state)
        if state.show_spectrum_marker:
            img = self.draw_spectrum_marker(img, state)
        return img
