from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image
import numpy as np

def generate_landslide_pdf(images, masks, overlays, filenames, areas_m2, areas_km2):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    margin_x = 2 * cm
    margin_y = 2 * cm
    vertical_gap = 0.5 * cm
    horizontal_gap = 0.5 * cm
    col_width = (page_width - 3 * margin_x) / 2
    row_height = (page_height - 4 * margin_y - 1.5 * vertical_gap) / 2

    for img, mask, overlay, name, m2, km2 in zip(images, masks, overlays, filenames, areas_m2, areas_km2):
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)

        if isinstance(mask, np.ndarray):
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            mask = Image.fromarray(mask.squeeze()).convert("L")

        if isinstance(overlay, np.ndarray):
            if overlay.dtype != np.uint8:
                overlay = (overlay * 255).astype(np.uint8)
            if overlay.ndim == 3 and overlay.shape[0] in [1, 3, 4]:
                overlay = np.transpose(overlay, (1, 2, 0))
            overlay = Image.fromarray(overlay)

        pdf.setFont("Helvetica-Bold", 14)
        filename_text = f"Filename: {name}"
        pdf.drawString((page_width - pdf.stringWidth(filename_text, "Helvetica-Bold", 14)) / 2,
                       page_height - margin_y, filename_text)

        pdf.setFont("Helvetica-Bold", 12)
        area_text = f"Landslide Area: {m2:.2f} m²"
        pdf.drawString((page_width - pdf.stringWidth(area_text, "Helvetica", 12)) / 2,
                       page_height - margin_y - 1.5 * vertical_gap, area_text)

        # Top row: RGB (left) and Mask (right)
        y_top = page_height - margin_y - 3 * vertical_gap - 0.5 * row_height
        for pil_img, x_pos, title in zip([img, mask], [margin_x, margin_x + col_width + horizontal_gap], ["Input Image", "Predicted Mask"]):
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(x_pos + (col_width - pdf.stringWidth(title, "Helvetica-Bold", 12)) / 2,
                           y_top + 0.5 * row_height, title)
            img_io = BytesIO()
            pil_img.save(img_io, format="PNG")
            img_io.seek(0)
            pdf.drawImage(ImageReader(img_io), x_pos, y_top - 0.5 * row_height,
                          width=col_width, height=row_height, preserveAspectRatio=True)

        # Bottom row: Overlay full width
        overlay_title_y = y_top - row_height - vertical_gap
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString((page_width - pdf.stringWidth("Overlay Image", "Helvetica-Bold", 12)) / 2,
                       overlay_title_y + 0.5 * row_height, "Overlay Image")
        overlay_io = BytesIO()
        overlay.save(overlay_io, format="PNG")
        overlay_io.seek(0)
        pdf.drawImage(ImageReader(overlay_io), margin_x, overlay_title_y - 0.5 * row_height,
                      width=page_width - 2 * margin_x, height=row_height, preserveAspectRatio=True)

        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer
