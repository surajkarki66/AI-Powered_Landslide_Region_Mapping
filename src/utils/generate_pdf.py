from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image

def generate_landslide_pdf(images, masks, filenames, areas_m2, areas_km2):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    margin_x = 2 * cm
    max_width = page_width - 2 * margin_x
    max_height = 8 * cm
    vertical_gap = 0.5 * cm  # gap between elements

    for img, mask, name, m2, km2 in zip(images, masks, filenames, areas_m2, areas_km2):
        # Filename and area info
        pdf.setFont("Helvetica-Bold", 14)
        filename_text = f"Filename: {name}"
        filename_width = pdf.stringWidth(filename_text, "Helvetica-Bold", 14)
        pdf.drawString((page_width - filename_width) / 2, page_height - 2 * cm, filename_text)

        pdf.setFont("Helvetica-Bold", 12)
        area_text = f"Landslide Area: {m2:.2f} m² ({km2:.4f} km²)"
        area_width = pdf.stringWidth(area_text, "Helvetica", 12)
        pdf.drawString((page_width - area_width) / 2, page_height - 2 * cm - 1.2 * vertical_gap, area_text)

        # Title for Input Image (centered)
        input_img_title_y = page_height - 2 * cm - 3 * vertical_gap
        pdf.setFont("Helvetica-Bold", 12)
        input_title = "Input Image"
        input_title_width = pdf.stringWidth(input_title, "Helvetica-Bold", 12)
        pdf.drawString((page_width - input_title_width) / 2, input_img_title_y, input_title)

        # Input Image
        image_io = BytesIO()
        img.save(image_io, format='PNG')
        image_io.seek(0)
        img_reader = ImageReader(image_io)
        img_y = input_img_title_y - max_height - vertical_gap / 2
        pdf.drawImage(img_reader, margin_x, img_y, width=max_width, height=max_height, preserveAspectRatio=True)

        # Title for Predicted Mask (centered)
        mask_title_y = img_y - vertical_gap - vertical_gap
        pdf.setFont("Helvetica-Bold", 12)
        mask_title = "Predicted Mask"
        mask_title_width = pdf.stringWidth(mask_title, "Helvetica-Bold", 12)
        pdf.drawString((page_width - mask_title_width) / 2, mask_title_y, mask_title)

        # Predicted Mask Image
        mask_img = Image.fromarray(mask.squeeze().astype('uint8')).convert('L')
        mask_io = BytesIO()
        mask_img.save(mask_io, format='PNG')
        mask_io.seek(0)
        mask_reader = ImageReader(mask_io)
        mask_y = mask_title_y - max_height - vertical_gap / 2
        pdf.drawImage(mask_reader, margin_x, mask_y, width=max_width, height=max_height, preserveAspectRatio=True)

        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer
